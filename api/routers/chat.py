from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from api.schemas.chat import ChatRequest, ChatResponse
from services.azure_cosmos import get_session_history, list_sessions
from services.azure_blob import get_source_sas_url
from agent.graph import app
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()


@router.get("/source-link")
async def get_source_link(
    file: str = Query(..., description="PDF blob name"),
    page: int = Query(1, description="Page number"),
):
    """Generate a short-lived SAS link to the PDF at a specific page."""
    try:
        sas_url = get_source_sas_url(
            blob_name=file,
            page_number=page,
        )
        return JSONResponse({"sas_url": sas_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def get_sessions_list():
    """Get list of recent session IDs for sidebar."""
    try:
        sessions = list_sessions()
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/{session_id}", response_model=ChatResponse)
async def get_chat_history_endpoint(session_id: str):
    try:
        items = get_session_history(session_id)
        formatted_history = []
        if items:
            for item in items:
                # Backwards compatible key checking
                u_content = item.get('user') or item.get('user_query')
                a_content = item.get('ai') or item.get('ai_response')
                
                if u_content:
                    formatted_history.append({"role": "user", "content": u_content})
                if a_content:
                    formatted_history.append({"role": "ai", "content": a_content})
                    
        return ChatResponse(
            response="History loaded.",
            session_id=session_id,
            chat_history=formatted_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        session_id = payload.session_id
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())

        user_input = payload.message
        
        # 1. Load History
        items = get_session_history(session_id)
        chat_history_objs = []
        
        if items:
            for item in items:
                u_content = item.get('user') or item.get('user_query')
                a_content = item.get('ai') or item.get('ai_response')
                
                if u_content:
                    chat_history_objs.append(HumanMessage(content=u_content))
                if a_content:
                    chat_history_objs.append(AIMessage(content=a_content))
        
        # 2. Invoke Agent
        result = app.invoke({
            "question": user_input,
            "chat_history": chat_history_objs,
            "retry_count": 0,
            "documents": [],
            "sources": [],
            "source_details": [],
            "session_id": session_id,
            "intent": ""
        })
        
        # 3. Extract final answer
        final_answer = result.get("final_answer", "No answer generated.")
        
        formatted_history = [
            {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else {"role": "ai", "content": msg.content}
            for msg in result.get("chat_history", [])
        ]

        return ChatResponse(
            response=final_answer,
            session_id=session_id,
            chat_history=formatted_history,
            source_details=result.get("source_details", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
