"""
agent.graph – LangGraph workflow definition
=============================================
Wires all nodes together into the final executable graph.

Flow:
    classify_intent
        ├─ greeting  → save_memory → END
        ├─ off_topic → save_memory → END
        └─ retrieve  → grade ─┬─ generate → save_memory → END
                               └─ retrieve  (retry, max 3)
"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    classify_intent_node,
    greeting_node,
    off_topic_node,
    retrieve_node,
    grade_documents_node,
    generate_node,
    save_memory_node,
)

# ── Build graph ──────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("greeting",        greeting_node)
workflow.add_node("off_topic",       off_topic_node)
workflow.add_node("retrieve",        retrieve_node)
workflow.add_node("grade",           grade_documents_node)
workflow.add_node("generate",        generate_node)
workflow.add_node("save_memory",     save_memory_node)

# ── Edges ────────────────────────────────────────────────────────────────

workflow.set_entry_point("classify_intent")


def _route_by_intent(state: AgentState) -> str:
    """Route to the correct branch based on classified intent."""
    intent = state.get("intent", "yolo")
    if intent == "greeting":
        return "greeting"
    if intent == "off_topic":
        return "off_topic"
    return "retrieve"


workflow.add_conditional_edges(
    "classify_intent",
    _route_by_intent,
    {"greeting": "greeting", "off_topic": "off_topic", "retrieve": "retrieve"},
)

# Non-RAG branches → straight to memory
workflow.add_edge("greeting",  "save_memory")
workflow.add_edge("off_topic", "save_memory")

# RAG pipeline
workflow.add_edge("retrieve", "grade")


def _decide_after_grading(state: AgentState) -> str:
    """Retry retrieval if documents were rejected (up to 3 times)."""
    if not state["documents"] and state["retry_count"] < 3:
        return "retrieve"
    return "generate"


workflow.add_conditional_edges(
    "grade",
    _decide_after_grading,
    {"retrieve": "retrieve", "generate": "generate"},
)

workflow.add_edge("generate",    "save_memory")
workflow.add_edge("save_memory", END)

# ── Compile ──────────────────────────────────────────────────────────────

app = workflow.compile()
