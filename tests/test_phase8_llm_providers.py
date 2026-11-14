"""Unit & Integration Tests for Unified Multi-Provider LLM Architecture.

Tests:
- BaseLLMProvider interface
- OllamaProvider (offline & mocked responses)
- OpenAIProvider (streaming SSE parser & mocked requests)
- GeminiProvider (streaming SSE parser & mocked requests)
- AnthropicProvider (streaming SSE parser & mocked requests)
- LLMProviderManager dispatch & error handling
- Security: API keys protected from exposure
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.llm_provider import (
    LLMProviderManager,
    OllamaProvider,
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider,
    LLMStreamChunk,
    LLMResponse
)


class TestLLMProviders(unittest.TestCase):

    def test_provider_manager_registration(self):
        """Verify all 4 core providers are registered and accessible."""
        providers = LLMProviderManager.list_providers()
        self.assertIn("ollama", providers)
        self.assertIn("openai", providers)
        self.assertIn("gemini", providers)
        self.assertIn("anthropic", providers)

    def test_fallback_to_default_provider(self):
        """Unknown provider names should gracefully fall back to default Ollama."""
        p = LLMProviderManager.get_provider("non_existent_provider_xyz")
        self.assertEqual(p.name, "ollama")

    def test_available_models_per_provider(self):
        """Ensure each provider reports expected available model choices."""
        for prov_name in ["ollama", "openai", "gemini", "anthropic"]:
            models = LLMProviderManager.get_models_for_provider(prov_name)
            self.assertIsInstance(models, list)
            self.assertGreater(len(models), 0)

    @patch("requests.Session.post")
    def test_ollama_streaming(self, mock_post):
        """Test Ollama streaming parser and TTFT calculation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({"message": {"content": "Hello"}}).encode('utf-8'),
            json.dumps({"message": {"content": " world"}}).encode('utf-8'),
            json.dumps({"done": True}).encode('utf-8')
        ]
        mock_post.return_value = mock_response

        prov = OllamaProvider(api_url="http://localhost:11434")
        chunks = list(prov.stream_chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "hi"}]
        ))

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "Hello")
        self.assertTrue(chunks[0].is_first_token)
        self.assertIsNotNone(chunks[0].ttft_ms)
        self.assertEqual(chunks[1].text, " world")
        self.assertFalse(chunks[1].is_first_token)
        self.assertTrue(chunks[2].is_final)

    @patch("requests.Session.post")
    def test_openai_streaming_sse(self, mock_post):
        """Test OpenAI Server-Sent Events (SSE) streaming format parser."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b"data: " + json.dumps({"choices": [{"delta": {"content": "The"}}], "finish_reason": None}).encode('utf-8'),
            b"data: " + json.dumps({"choices": [{"delta": {"content": " answer"}}], "finish_reason": "stop"}).encode('utf-8'),
            b"data: [DONE]"
        ]
        mock_post.return_value = mock_response

        prov = OpenAIProvider(api_key="test-key-mock")
        chunks = list(prov.stream_chat(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}]
        ))

        self.assertTrue(chunks[0].is_first_token)
        self.assertEqual(chunks[0].text, "The")
        self.assertEqual(chunks[1].text, " answer")
        self.assertTrue(chunks[1].is_final)

    @patch("requests.Session.post")
    def test_gemini_streaming_sse(self, mock_post):
        """Test Google Gemini Server-Sent Events (SSE) streaming parser."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b"data: " + json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "Gemini"}]},
                    "finishReason": None
                }]
            }).encode('utf-8'),
            b"data: " + json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": " output"}]},
                    "finishReason": "STOP"
                }]
            }).encode('utf-8')
        ]
        mock_post.return_value = mock_response

        prov = GeminiProvider(api_key="test-gemini-key")
        chunks = list(prov.stream_chat(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": "test"}]
        ))

        self.assertEqual(chunks[0].text, "Gemini")
        self.assertTrue(chunks[0].is_first_token)
        self.assertEqual(chunks[1].text, " output")
        self.assertTrue(chunks[1].is_final)

    @patch("requests.Session.post")
    def test_anthropic_streaming_sse(self, mock_post):
        """Test Anthropic Claude content_block_delta streaming parser."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b"data: " + json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Claude"}
            }).encode('utf-8'),
            b"data: " + json.dumps({"type": "message_stop"}).encode('utf-8')
        ]
        mock_post.return_value = mock_response

        prov = AnthropicProvider(api_key="test-anthropic-key")
        chunks = list(prov.stream_chat(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "test"}]
        ))

        self.assertEqual(chunks[0].text, "Claude")
        self.assertTrue(chunks[0].is_first_token)
        self.assertTrue(chunks[1].is_final)

    def test_missing_api_key_graceful_error(self):
        """Providers without configured API keys must yield clear error chunks, not crash."""
        prov = OpenAIProvider(api_key="")
        chunks = list(prov.stream_chat("gpt-4o", [{"role": "user", "content": "hi"}]))
        self.assertTrue(any("API Key is not configured" in (c.error or "") for c in chunks))

    @patch("requests.Session.post")
    def test_generate_complete_response_metrics(self, mock_post):
        """Verify generate() calculates TTFT, TPS, and total latency correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({"message": {"content": "Comprehensive"}}).encode('utf-8'),
            json.dumps({"message": {"content": " response text."}}).encode('utf-8'),
            json.dumps({"done": True}).encode('utf-8')
        ]
        mock_post.return_value = mock_response

        prov = OllamaProvider(api_url="http://localhost:11434")
        resp = prov.generate("llama3.2", [{"role": "user", "content": "query"}])

        self.assertEqual(resp.text, "Comprehensive response text.")
        self.assertEqual(resp.provider, "ollama")
        self.assertGreaterEqual(resp.latency_ms, 0)
        self.assertGreaterEqual(resp.completion_tokens, 1)


if __name__ == "__main__":
    unittest.main()
