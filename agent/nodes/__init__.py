"""
agent.nodes – Node sub-package
================================
Shared service clients are initialized here once and imported 
by individual node modules to avoid duplicate connections.

Re-exports all node functions for clean imports in graph.py.
"""

from services.azure_openai import get_chat_llm, get_embedding_client
from services.azure_search import get_search_client
from services.azure_cosmos import get_chat_container

# ── Shared clients (initialized once, used by all nodes) ─────────────────
llm = get_chat_llm()
search_client = get_search_client()
embedding_client = get_embedding_client()
cosmos_container = get_chat_container()

# ── Re-export all node functions ─────────────────────────────────────────
from agent.nodes.intent import classify_intent_node, greeting_node, off_topic_node
from agent.nodes.retriever import retrieve_node
from agent.nodes.grader import grade_documents_node
from agent.nodes.generator import generate_node
from agent.nodes.memory import save_memory_node

__all__ = [
    "classify_intent_node",
    "greeting_node",
    "off_topic_node",
    "retrieve_node",
    "grade_documents_node",
    "generate_node",
    "save_memory_node",
]
