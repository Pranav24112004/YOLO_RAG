from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Union

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class SourceDetail(BaseModel):
    file: str
    page: Union[str, int] = "?"
    section: str = "General"
    snippet: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    chat_history: Optional[List[dict]] = None
    source_details: Optional[List[SourceDetail]] = None

    model_config = ConfigDict(from_attributes=True)
