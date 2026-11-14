import requests
import json
import logging
import urllib.parse
import re
from typing import List, Dict, Any, Generator, Tuple, Optional
import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.llm_provider import LLMProviderManager, LLMStreamChunk
from utils.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class WebSearchManager:
    @staticmethod
    def search_duckduckgo(query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """Perform a web search using DuckDuckGo HTML search and extract titles, urls, and snippets."""
        results = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            response = requests.get(url, headers=headers, timeout=8)

            if response.status_code != 200:
                logger.warning(f"DuckDuckGo search failed with status {response.status_code}")
                return results

            html = response.text
            containers = re.findall(r'<div class="result.*?">(.*?)</div>\s*</div>', html, re.DOTALL)

            for container in containers[:max_results]:
                link_match = re.search(r'<a class="result__url" href="(.*?)".*?>(.*?)</a>', container, re.DOTALL)
                snippet_match = re.search(r'<a class="result__snippet".*?>(.*?)</a>', container, re.DOTALL)

                if link_match:
                    href = link_match.group(1).strip()
                    if "declined_uddg=" in href:
                        parsed = urllib.parse.urlparse(href)
                        queries = urllib.parse.parse_qs(parsed.query)
                        href = queries.get("uddg", [href])[0]

                    title = re.sub(r'<.*?>', '', link_match.group(2)).strip()
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r'<.*?>', '', snippet_match.group(1)).strip()

                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": snippet
                    })
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")

        return results


class PromptGroundingManager:
    @staticmethod
    def format_grounded_rag_prompt(
        query: str, 
        rag_chunks: List[Dict[str, Any]], 
        personality_prompt: str,
        context_budget_tokens: int = 2048,
        template_name: str = "question_answering",
        conversation_context: str = ""
    ) -> Tuple[str, str, List[str]]:
        """Construct a strictly grounded prompt with citations, context budget limits, and prompt-injection defenses.
        Delegates to centralized PromptManager while preserving full backwards compatibility.
        """
        return PromptManager.format_grounded_prompt(
            template_name=template_name,
            query=query,
            rag_chunks=rag_chunks,
            personality_prompt=personality_prompt,
            context_budget_tokens=context_budget_tokens,
            conversation_context=conversation_context
        )


class OllamaChatManager:
    @staticmethod
    def get_available_models() -> List[str]:
        """Fetch list of models available in the local Ollama instance."""
        return LLMProviderManager.get_models_for_provider("ollama")

    @staticmethod
    def generate_chat_response(
        model: str,
        messages: List[Dict[str, str]],
        settings: Dict[str, Any] = None,
        provider: str = "ollama"
    ) -> Generator[str, None, None]:
        """Call LLM Provider with persistent session, streaming, and cancellation support."""
        settings = settings or {}
        for chunk in LLMProviderManager.stream_chat(provider, model, messages, settings):
            if st.session_state.get("stop_generation", False):
                break
            if chunk.error:
                yield f"\n⚠️ **Error:** {chunk.error}"
                break
            if chunk.text:
                yield chunk.text
