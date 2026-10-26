from fastapi import FastAPI
from api.routers import chat
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="YOLO RAG API",
    description="API for the YOLO Research Assistant Chatbot",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to YOLO RAG API. Visit /docs for Swagger UI."}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
