"""
services.azure_openai  →  Azure OpenAI wrapper
================================================
Provides embedding and chat LLM clients backed by Azure OpenAI
(text-embedding-3-small + GPT-4o-mini / o4-mini).
"""

import requests
from langchain_openai import AzureChatOpenAI
import config.settings as settings


# ---------------------------------------------------------------------------
# Embedding helper  (lightweight – calls the REST endpoint directly)
# ---------------------------------------------------------------------------

class _AzureEmbeddingClient:
    """Minimal wrapper that exposes an .embeddings.create() interface
    calling Azure OpenAI text-embedding-3-small."""

    def __init__(self):
        self.embeddings = self

    def create(self, *, model: str = None, inputs: list[str] = None, input: list[str] = None):
        """
        Accepts either 'inputs' or 'input' (list of strings).
        Returns an object with .data[0].embedding  (list[float]).
        """
        texts = inputs or input
        if texts is None:
            raise ValueError("Must provide 'inputs' or 'input'")

        # The endpoint env var already contains the full deployment URL
        url = settings.AZURE_TEXT_EMBEDDING_ENDPOINT
        headers = {
            "Content-Type": "application/json",
            "api-key": settings.AZURE_TEXT_EMBEDDING_KEY,
        }
        body = {"input": texts}
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Wrap in a tiny object so callers can keep using .data[0].embedding
        class _EmbItem:
            def __init__(self, vec):
                self.embedding = vec

        class _EmbResponse:
            def __init__(self, items):
                self.data = items

        return _EmbResponse([_EmbItem(d["embedding"]) for d in data["data"]])


def get_embedding_client():
    """Returns an object with .embeddings.create() for Azure OpenAI embeddings."""
    return _AzureEmbeddingClient()


# ---------------------------------------------------------------------------
# Chat LLM  (LangChain AzureChatOpenAI wrapping GPT-4o-mini)
# ---------------------------------------------------------------------------

def get_chat_llm(temperature=None):
    """
    Returns an AzureChatOpenAI instance for GPT-4o-mini (o4-mini).
    Note: o4-mini does NOT support custom temperature — only the default (1).
    The temperature parameter is accepted but ignored for compatibility.
    """
    full_url = settings.AZURE_GPT_4O_MINI_ENDPOINT
    # AzureChatOpenAI needs just the host part (everything before /openai/)
    base_endpoint = full_url.split("/openai/")[0]

    return AzureChatOpenAI(
        azure_endpoint=base_endpoint,
        azure_deployment=settings.LLM_DEPLOYMENT,
        api_key=settings.AZURE_GPT_4O_MINI_KEY,
        api_version=settings.LLM_API_VERSION,
    )
