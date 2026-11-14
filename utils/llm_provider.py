"""Unified Multi-Provider LLM Architecture for Corpus Mind.

Supports:
- Local Models: Ollama (REST API with streaming & persistent sessions)
- Cloud Models: OpenAI (and OpenAI-compatible), Google Gemini, Anthropic Claude
- Unified streaming, metrics (TTFT, TPS, latency), error recovery, and structured JSON generation.
"""

import os
import sys
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Tuple, Optional, Type
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.structured_output import StructuredOutputParser

logger = logging.getLogger(__name__)


# =====================================================================
# DATA CLASSES & METRICS
# =====================================================================

@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMStreamChunk:
    text: str
    is_first_token: bool = False
    is_final: bool = False
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    ttft_ms: Optional[float] = None
    elapsed_ms: Optional[float] = None


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = "stop"
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


# =====================================================================
# BASE PROVIDER INTERFACE
# =====================================================================

class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM Providers."""

    def __init__(self, name: str):
        self.name = name
        self._session = self._create_resilient_session()

    @staticmethod
    def _create_resilient_session() -> requests.Session:
        """Create a requests session with connection pooling and keep-alive."""
        session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Test provider connectivity and credentials. Returns (is_ok, message)."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return list of models supported or available for this provider."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        """Stream response tokens progressively with latency and TTFT tracking."""
        pass

    def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate a complete response by consuming the streaming generator."""
        start_time = time.time()
        full_text = []
        ttft = 0.0
        error = None
        finish_reason = "stop"

        for chunk in self.stream_chat(model, messages, settings):
            if chunk.error:
                error = chunk.error
                break
            if chunk.is_first_token and chunk.ttft_ms:
                ttft = chunk.ttft_ms
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.text:
                full_text.append(chunk.text)

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000.0
        response_text = "".join(full_text)
        
        # Estimate token count (rough heuristic: 1 token ~ 4 characters)
        comp_tokens = max(1, len(response_text) // 4)
        tps = (comp_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

        return LLMResponse(
            text=response_text,
            model=model,
            provider=self.name,
            completion_tokens=comp_tokens,
            total_tokens=comp_tokens,
            latency_ms=round(latency_ms, 2),
            ttft_ms=round(ttft, 2),
            tokens_per_second=round(tps, 2),
            finish_reason=finish_reason,
            error=error
        )

    def generate_structured(
        self,
        model: str,
        messages: List[Dict[str, str]],
        schema_class: Type[BaseModel],
        settings: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> Tuple[Optional[BaseModel], str, Dict[str, Any]]:
        """Generate a structured response adhering to schema_class with automatic repair."""
        settings = settings or {}
        # Append schema instructions to user prompt
        schema_instruction = StructuredOutputParser.get_schema_json_instructions(schema_class)
        
        modified_messages = [dict(m) for m in messages]
        if modified_messages and modified_messages[-1]["role"] == "user":
            modified_messages[-1]["content"] += f"\n\n{schema_instruction}"
        else:
            modified_messages.append({"role": "user", "content": schema_instruction})

        attempt = 0
        last_error = None
        last_raw_text = ""

        while attempt <= max_retries:
            resp = self.generate(model, modified_messages, settings)
            last_raw_text = resp.text
            if resp.error:
                return None, resp.error, {}

            instance, err, raw_dict = StructuredOutputParser.validate_and_parse(resp.text, schema_class)
            if instance is not None:
                return instance, "", raw_dict or {}

            last_error = err
            logger.warning(f"Structured output retry {attempt + 1}/{max_retries} due to: {err}")
            attempt += 1

            if attempt <= max_retries:
                # Add correction message to prompt
                correction_prompt = (
                    f"Your previous output failed schema validation: {err}.\n"
                    "Please correct the output and respond ONLY with the repaired valid JSON object."
                )
                modified_messages.append({"role": "assistant", "content": resp.text})
                modified_messages.append({"role": "user", "content": correction_prompt})

        return None, f"Failed after {max_retries} retries: {last_error}. Raw response: {last_raw_text[:200]}", {}


# =====================================================================
# 1. OLLAMA PROVIDER (LOCAL)
# =====================================================================

class OllamaProvider(BaseLLMProvider):
    """Local LLM Provider via Ollama REST API with native streaming."""

    def __init__(self, api_url: Optional[str] = None):
        super().__init__(name="ollama")
        self.api_url = (api_url or getattr(config, "OLLAMA_API_URL", "http://localhost:11434")).rstrip("/")

    def test_connection(self) -> Tuple[bool, str]:
        try:
            url = f"{self.api_url}/api/tags"
            resp = self._session.get(url, timeout=4)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                model_str = f" ({len(models)} models available: {', '.join(models[:3])}...)" if models else ""
                return True, f"Connected to Ollama service{model_str}"
            return False, f"Ollama returned HTTP status {resp.status_code}"
        except Exception as e:
            return False, f"Cannot reach Ollama at {self.api_url}: {str(e)}"

    def get_available_models(self) -> List[str]:
        try:
            url = f"{self.api_url}/api/tags"
            resp = self._session.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m['name'] for m in data.get('models', [])]
                if models:
                    return models
        except Exception:
            pass
        return ["llama3.2:latest", "llama3.2", "phi3", "mistral", "qwen2.5"]

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        settings = settings or {}
        temperature = settings.get("temperature", config.DEFAULT_TEMPERATURE)
        max_tokens = settings.get("max_tokens", config.DEFAULT_MAX_TOKENS)
        top_p = settings.get("top_p", 0.9)
        context_len = settings.get("context_length", 2048)
        json_mode = settings.get("json_mode", False)
        timeout = settings.get("timeout", config.LLM_REQUEST_TIMEOUT)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "num_ctx": context_len
            }
        }
        if json_mode:
            payload["format"] = "json"

        start_time = time.time()
        first_token_seen = False
        url = f"{self.api_url}/api/chat"

        try:
            resp = self._session.post(url, json=payload, stream=True, timeout=(6, timeout))
            if resp.status_code != 200:
                yield LLMStreamChunk(
                    text="",
                    error=f"Ollama error {resp.status_code}: {resp.text[:200]}"
                )
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    msg = chunk.get('message', {})
                    token = msg.get('content', '')
                    done = chunk.get('done', False)

                    ttft_val = None
                    is_first = False
                    if token and not first_token_seen:
                        first_token_seen = True
                        is_first = True
                        ttft_val = round((time.time() - start_time) * 1000.0, 2)

                    yield LLMStreamChunk(
                        text=token,
                        is_first_token=is_first,
                        is_final=done,
                        ttft_ms=ttft_val,
                        elapsed_ms=round((time.time() - start_time) * 1000.0, 2)
                    )
                except Exception as parse_err:
                    logger.debug(f"Ollama stream parse error: {parse_err}")

        except requests.exceptions.Timeout:
            yield LLMStreamChunk(text="", error=f"Ollama connection timed out at {self.api_url}")
        except Exception as e:
            yield LLMStreamChunk(text="", error=f"Ollama connection error: {str(e)}")


# =====================================================================
# 2. OPENAI PROVIDER (CLOUD / OPENAI-COMPATIBLE)
# =====================================================================

class OpenAIProvider(BaseLLMProvider):
    """OpenAI and OpenAI-compatible Cloud Provider (e.g. OpenAI, Groq, OpenRouter)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(name="openai")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

    def test_connection(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "OpenAI API Key is missing. Please configure OPENAI_API_KEY in Settings."
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = self._session.get(f"{self.base_url}/models", headers=headers, timeout=5)
            if resp.status_code == 200:
                return True, "Connected to OpenAI API successfully"
            elif resp.status_code == 401:
                return False, "Authentication failed: Invalid OpenAI API Key"
            return False, f"OpenAI API responded with HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Cannot connect to OpenAI API: {str(e)}"

    def get_available_models(self) -> List[str]:
        default_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        if not self.api_key:
            return default_models
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = self._session.get(f"{self.base_url}/models", headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", []) if "gpt" in m.get("id", "").lower()]
                if models:
                    return sorted(models)
        except Exception:
            pass
        return default_models

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        if not self.api_key:
            yield LLMStreamChunk(text="", error="OpenAI API Key is not configured. Set OPENAI_API_KEY in Settings.")
            return

        settings = settings or {}
        temperature = settings.get("temperature", config.DEFAULT_TEMPERATURE)
        max_tokens = settings.get("max_tokens", config.DEFAULT_MAX_TOKENS)
        top_p = settings.get("top_p", 1.0)
        json_mode = settings.get("json_mode", False)
        timeout = settings.get("timeout", config.LLM_REQUEST_TIMEOUT)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start_time = time.time()
        first_token_seen = False

        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=(6, timeout)
            )
            if resp.status_code != 200:
                yield LLMStreamChunk(text="", error=f"OpenAI error {resp.status_code}: {resp.text[:200]}")
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    line_data = line_str[6:].strip()
                    if line_data == "[DONE]":
                        yield LLMStreamChunk(text="", is_final=True)
                        break
                    try:
                        chunk_obj = json.loads(line_data)
                        choices = chunk_obj.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            token = delta.get("content", "")
                            finish_reason = choices[0].get("finish_reason") or chunk_obj.get("finish_reason")

                            ttft_val = None
                            is_first = False
                            if token and not first_token_seen:
                                first_token_seen = True
                                is_first = True
                                ttft_val = round((time.time() - start_time) * 1000.0, 2)

                            yield LLMStreamChunk(
                                text=token,
                                is_first_token=is_first,
                                is_final=(finish_reason is not None),
                                finish_reason=finish_reason,
                                ttft_ms=ttft_val,
                                elapsed_ms=round((time.time() - start_time) * 1000.0, 2)
                            )
                    except Exception:
                        pass
        except Exception as e:
            yield LLMStreamChunk(text="", error=f"OpenAI streaming error: {str(e)}")


# =====================================================================
# 3. GOOGLE GEMINI PROVIDER (CLOUD)
# =====================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API Provider with streaming SSE and JSON mode support."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="gemini")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    def test_connection(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini API Key is missing. Configure GEMINI_API_KEY in Settings."
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            resp = self._session.get(url, timeout=5)
            if resp.status_code == 200:
                return True, "Connected to Google Gemini API successfully"
            elif resp.status_code == 400 or resp.status_code == 403:
                return False, f"Gemini API Key invalid or permissions error (HTTP {resp.status_code})"
            return False, f"Gemini API returned HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Cannot reach Gemini API: {str(e)}"

    def get_available_models(self) -> List[str]:
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-1.0-pro"]

    def _convert_messages(self, messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert standard messages into Gemini systemInstruction + contents structure."""
        system_instruction = None
        contents = []

        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})

        return system_instruction, contents

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        if not self.api_key:
            yield LLMStreamChunk(text="", error="Gemini API Key is not configured. Set GEMINI_API_KEY in Settings.")
            return

        settings = settings or {}
        temperature = settings.get("temperature", config.DEFAULT_TEMPERATURE)
        max_tokens = settings.get("max_tokens", config.DEFAULT_MAX_TOKENS)
        top_p = settings.get("top_p", 0.95)
        json_mode = settings.get("json_mode", False)
        timeout = settings.get("timeout", config.LLM_REQUEST_TIMEOUT)

        sys_inst, contents = self._convert_messages(messages)
        gen_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": top_p
        }
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_config
        }
        if sys_inst:
            payload["systemInstruction"] = {"parts": [{"text": sys_inst}]}

        clean_model = model.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:streamGenerateContent?key={self.api_key}&alt=sse"

        start_time = time.time()
        first_token_seen = False

        try:
            headers = {"Content-Type": "application/json"}
            resp = self._session.post(url, headers=headers, json=payload, stream=True, timeout=(6, timeout))
            if resp.status_code != 200:
                yield LLMStreamChunk(text="", error=f"Gemini API error {resp.status_code}: {resp.text[:200]}")
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    line_data = line_str[6:].strip()
                    try:
                        chunk_obj = json.loads(line_data)
                        candidates = chunk_obj.get("candidates", [])
                        if candidates:
                            cand = candidates[0]
                            parts = cand.get("content", {}).get("parts", [])
                            finish_reason = cand.get("finishReason")
                            for part in parts:
                                token = part.get("text", "")
                                ttft_val = None
                                is_first = False
                                if token and not first_token_seen:
                                    first_token_seen = True
                                    is_first = True
                                    ttft_val = round((time.time() - start_time) * 1000.0, 2)

                                yield LLMStreamChunk(
                                    text=token,
                                    is_first_token=is_first,
                                    is_final=(finish_reason is not None),
                                    finish_reason=finish_reason,
                                    ttft_ms=ttft_val,
                                    elapsed_ms=round((time.time() - start_time) * 1000.0, 2)
                                )
                    except Exception:
                        pass
        except Exception as e:
            yield LLMStreamChunk(text="", error=f"Gemini streaming error: {str(e)}")


# =====================================================================
# 4. ANTHROPIC CLAUDE PROVIDER (CLOUD)
# =====================================================================

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Cloud Provider using the Messages API."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="anthropic")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def test_connection(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "Anthropic API Key is missing. Configure ANTHROPIC_API_KEY in Settings."
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            # Test with a minimal 1-token query
            payload = {
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]
            }
            resp = self._session.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=5)
            if resp.status_code == 200:
                return True, "Connected to Anthropic API successfully"
            elif resp.status_code == 401:
                return False, "Invalid Anthropic API Key"
            return False, f"Anthropic API returned HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Cannot connect to Anthropic API: {str(e)}"

    def get_available_models(self) -> List[str]:
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229"
        ]

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        if not self.api_key:
            yield LLMStreamChunk(text="", error="Anthropic API Key is not configured. Set ANTHROPIC_API_KEY in Settings.")
            return

        settings = settings or {}
        temperature = settings.get("temperature", config.DEFAULT_TEMPERATURE)
        max_tokens = settings.get("max_tokens", config.DEFAULT_MAX_TOKENS)
        timeout = settings.get("timeout", config.LLM_REQUEST_TIMEOUT)

        system_prompt = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += m["content"] + "\n\n"
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        if system_prompt.strip():
            payload["system"] = system_prompt.strip()

        start_time = time.time()
        first_token_seen = False

        try:
            resp = self._session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                stream=True,
                timeout=(6, timeout)
            )
            if resp.status_code != 200:
                yield LLMStreamChunk(text="", error=f"Anthropic error {resp.status_code}: {resp.text[:200]}")
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    line_data = line_str[6:].strip()
                    try:
                        event_data = json.loads(line_data)
                        event_type = event_data.get("type")
                        if event_type == "content_block_delta":
                            token = event_data.get("delta", {}).get("text", "")
                            ttft_val = None
                            is_first = False
                            if token and not first_token_seen:
                                first_token_seen = True
                                is_first = True
                                ttft_val = round((time.time() - start_time) * 1000.0, 2)

                            yield LLMStreamChunk(
                                text=token,
                                is_first_token=is_first,
                                ttft_ms=ttft_val,
                                elapsed_ms=round((time.time() - start_time) * 1000.0, 2)
                            )
                        elif event_type == "message_stop":
                            yield LLMStreamChunk(text="", is_final=True)
                    except Exception:
                        pass
        except Exception as e:
            yield LLMStreamChunk(text="", error=f"Anthropic streaming error: {str(e)}")


# =====================================================================
# 5. UNIFIED LLM PROVIDER MANAGER
# =====================================================================

class LLMProviderManager:
    """Singleton Manager dispatching requests to configured local or cloud LLM providers."""

    _providers: Dict[str, BaseLLMProvider] = {}
    _initialized = False

    @classmethod
    def _init_providers(cls):
        if not cls._initialized:
            cls._providers = {
                "ollama": OllamaProvider(),
                "openai": OpenAIProvider(),
                "gemini": GeminiProvider(),
                "anthropic": AnthropicProvider()
            }
            cls._initialized = True

    @classmethod
    def get_provider(cls, name: str) -> BaseLLMProvider:
        cls._init_providers()
        provider = cls._providers.get(name.lower())
        if not provider:
            logger.warning(f"Unknown provider '{name}', falling back to Ollama.")
            provider = cls._providers["ollama"]
        return provider

    @classmethod
    def list_providers(cls) -> List[str]:
        cls._init_providers()
        return list(cls._providers.keys())

    @classmethod
    def test_all_providers(cls) -> Dict[str, Tuple[bool, str]]:
        cls._init_providers()
        return {name: p.test_connection() for name, p in cls._providers.items()}

    @classmethod
    def get_models_for_provider(cls, provider_name: str) -> List[str]:
        return cls.get_provider(provider_name).get_available_models()

    @classmethod
    def stream_chat(
        cls,
        provider_name: str,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> Generator[LLMStreamChunk, None, None]:
        provider = cls.get_provider(provider_name)
        return provider.stream_chat(model, messages, settings)

    @classmethod
    def generate(
        cls,
        provider_name: str,
        model: str,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        provider = cls.get_provider(provider_name)
        return provider.generate(model, messages, settings)

    @classmethod
    def generate_structured(
        cls,
        provider_name: str,
        model: str,
        messages: List[Dict[str, str]],
        schema_class: Type[BaseModel],
        settings: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[BaseModel], str, Dict[str, Any]]:
        provider = cls.get_provider(provider_name)
        return provider.generate_structured(model, messages, schema_class, settings)
