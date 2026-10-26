"""
agent.nodes.generator – RAG answer generation with citations
"""

from langchain_core.messages import HumanMessage, AIMessage

from agent.state import AgentState
from agent import prompts
from core.logger import log_event


def generate_node(state: AgentState):
    """Generate a cited answer from retrieved documents."""
    from agent.nodes import llm

    print("     [Writer]: Generating answer...")
    docs     = state["documents"]
    history  = state["chat_history"][-4:]
    question = state["question"]

    prompt = prompts.GENERATE_ANSWER.format(
        context="\n\n".join(docs),
        history=str(history),
        question=question,
    )
    response = llm.invoke(prompt)

    updated_hist = state["chat_history"] + [
        HumanMessage(content=question),
        AIMessage(content=response.content),
    ]
    log_event(
        state.get("session_id"), 
        "GENERATION", 
        {"answer_length": len(response.content)}
    )

    return {
        "final_answer": response.content,
        "chat_history": updated_hist,
        "source_details": state.get("source_details", []),
    }
