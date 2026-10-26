
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    documents: List[str]
    retry_count: int
    final_answer: str
    session_id: str
    sources: List[str]
    source_details: List[dict]
    intent: str  # "greeting", "yolo", or "off_topic"
