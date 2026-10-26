"""
agent.nodes.memory – Persist conversation turns to Cosmos DB
"""

import uuid
from datetime import datetime

from agent.state import AgentState
from core.logger import log_event


from azure.cosmos.exceptions import CosmosResourceNotFoundError

def save_memory_node(state: AgentState):
    """Save the latest user/AI turn to the consolidated session document in Cosmos DB."""
    from agent.nodes import cosmos_container

    session_id = state.get("session_id")
    if not session_id:
        return
    
    if len(state["chat_history"]) < 2:
        return {"final_answer": state.get("final_answer", "")}

    last_user = state["chat_history"][-2]
    last_ai   = state["chat_history"][-1]
    
    # Try to fetch existing session document
    try:
        doc = cosmos_container.read_item(item=session_id, partition_key=session_id)
    except CosmosResourceNotFoundError:
        # Initialize new session document
        doc = {
            "id": session_id,
            "session_id": session_id,
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }
    
    # Append the new turn
    new_turn = {
        "user": last_user.content,
        "ai":   last_ai.content
    }
    doc["messages"].append(new_turn)
    doc["last_updated"] = datetime.now().isoformat()
    
    # Upsert the entire document
    cosmos_container.upsert_item(body=doc)
    
    log_event(
        session_id, 
        "SESSION_SAVED", 
        {"message_count": len(doc["messages"])}
    )
    
    return {"final_answer": state["final_answer"]}
