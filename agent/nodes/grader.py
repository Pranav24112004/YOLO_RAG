"""
agent.nodes.grader – Document relevance grading & query rewriting
"""

from agent.state import AgentState
from agent import prompts
from core.logger import log_event


def grade_documents_node(state: AgentState):
    """Grade retrieved documents for relevance; rewrite query if rejected."""
    from agent.nodes import llm

    print("     [Critic]: Grading documents...")
    question = state["question"]
    docs = state["documents"]

    prompt = prompts.GRADE_DOCUMENTS.format(question=question, documents=str(docs))
    response = llm.invoke(prompt).content.strip().upper()

    if "YES" in response:
        print("[OK] [Critic]: Documents approved.")
        log_event(state.get("session_id"), "GRADING", {"status": "APPROVED", "retry_count": state.get("retry_count", 0)})
        return {"retry_count": state.get("retry_count", 0)}

    # --- Rewrite query on rejection ---
    print("[FAIL] [Critic]: Documents rejected. Rewriting query...")
    rewrite_prompt = prompts.REWRITE_QUERY.format(question=question)
    new_query = llm.invoke(rewrite_prompt).content.strip()
    print(f"[Retry] [System]: New Query: '{new_query}'")

    log_event(
        state.get("session_id"), 
        "GRADING", 
        {"status": "REJECTED", "new_query": new_query, "retry_count": state.get("retry_count", 0) + 1}
    )

    return {
        "documents": [],
        "question": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }
