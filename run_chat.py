
from agent.graph import app
from services.azure_cosmos import get_session_history
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import hashlib

def main():
    print("🤖 YOLO Agent (Refactored)")
    
    mode = input("Start new session (n) or continue existing (c)? [n]: ").strip().lower()
    
    if mode == 'c':
        session_id = input("Enter Session ID to continue: ").strip()
        print(f"🔄 Loading history for {session_id}...")
        try:
            items = get_session_history(session_id)
            if not items:
                print("⚠️ No history found. Starting fresh.")
                chat_history = []
            else:
                # Convert DB items to LangChain messages
                chat_history = []
                for turn in items:
                    if 'user' in turn:
                        chat_history.append(HumanMessage(content=turn['user']))
                    if 'ai' in turn:
                        chat_history.append(AIMessage(content=turn['ai']))
                print(f"✅ Loaded {len(items)} exchanges.")
        except Exception as e:
            print(f"❌ Error loading history: {e}")
            chat_history = []
            session_str = datetime.now().isoformat()
            session_id = hashlib.sha256(session_str.encode()).hexdigest()[:16]
    else:
        # New Session
        session_str = datetime.now().isoformat()
        session_id = hashlib.sha256(session_str.encode()).hexdigest()[:16]
        chat_history = []
    
    print(f"Session: {session_id}")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            result = app.invoke({
                "question": user_input,
                "chat_history": chat_history,
                "retry_count": 0,
                "documents": [],
                "session_id": session_id
            })
            
            print(f"🤖 Agent: {result['final_answer']}")
            
            # Print Sources with snippets and PDF links
            if 'source_details' in result and result['source_details']:
                print("\nSources Used:")
                print("-" * 60)
                for i, src in enumerate(result['source_details'], 1):
                    print(f"  [{i}] {src['file']} (Page {src['page']}) [{src['section']}]")
                    if src.get('snippet'):
                        print(f"      \"{src['snippet']}\"")
                    if src.get('pdf_url'):
                        print(f"      PDF: {src['pdf_url']}")
                    print()
            
            # Update history from result to keep in sync
            chat_history = result["chat_history"]

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
