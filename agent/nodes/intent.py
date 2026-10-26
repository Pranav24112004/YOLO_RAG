"""
agent.nodes.intent – Intent classification, greeting & off-topic handling
"""

from langchain_core.messages import HumanMessage, AIMessage

from agent.state import AgentState
from agent import prompts
from core.logger import log_event


def classify_intent_node(state: AgentState):
    """Classify the user's message as greeting, yolo-related, or off-topic."""
    from agent.nodes import llm

    question = state["question"]
    print(f"[Router]: Classifying intent for '{question}'...")

    # Include recent chat history so the classifier can understand follow-up questions
    recent_history = state.get("chat_history", [])[-4:]  # last 2 exchanges
    history_str = "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
        for msg in recent_history
    ) if recent_history else "(no prior conversation)"

    prompt = prompts.CLASSIFY_INTENT.format(chat_history=history_str, question=question)
    response = llm.invoke(prompt).content.strip().upper()

    if "GREETING" in response:
        intent = "greeting"
    elif "YOLO" in response:
        intent = "yolo"
    else:
        intent = "off_topic"

    print(f"[Router]: Intent = {intent}")
    
    log_event(
        state.get("session_id"), 
        "INTENT_CLASSIFICATION", 
        {"question": question, "intent": intent}
    )
    
    return {"intent": intent}


def greeting_node(state: AgentState):
    """Generate a friendly greeting response."""
    from agent.nodes import llm

    question = state["question"]
    print("[Greeter]: Responding to greeting...")

    prompt = prompts.GREETING_RESPONSE.format(question=question)
    response = llm.invoke(prompt)

    updated_hist = state["chat_history"] + [
        HumanMessage(content=question),
        AIMessage(content=response.content),
    ]
    
    log_event(
        state.get("session_id"), 
        "GREETING_RESPONSE", 
        {"response": response.content}
    )
    
    return {"final_answer": response.content, "chat_history": updated_hist}


def off_topic_node(state: AgentState):
    """Respond to off-topic messages with a polite redirect."""
    question = state["question"]
    print("[Router]: Off-topic query detected.")

    reply = prompts.OFF_TOPIC_RESPONSE

    updated_hist = state["chat_history"] + [
        HumanMessage(content=question),
        AIMessage(content=reply),
    ]
    
    log_event(
        state.get("session_id"), 
        "OFF_TOPIC_RESPONSE", 
        {"response": reply}
    )
    
    return {"final_answer": reply, "chat_history": updated_hist}
