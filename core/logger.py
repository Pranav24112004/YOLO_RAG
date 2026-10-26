import json
import os
from datetime import datetime

# Path to the logs directory
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

def log_event(session_id: str, step_name: str, details: dict = None):
    """
    Appends a step log to a session-specific JSON file.
    """
    if not session_id:
        session_id = "unknown_session"
        
    log_file = os.path.join(LOGS_DIR, f"{session_id}.json")
    
    # Structure of the log entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "details": details or {}
    }
    
    # Read existing context or start new list
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
            
    # Add new entry
    logs.append(entry)
    
    # Write back
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

def clear_logs(session_id: str):
    """Clears logs for a specific session (useful at start of new query)."""
    log_file = os.path.join(LOGS_DIR, f"{session_id}.json")
    if os.path.exists(log_file):
        os.remove(log_file)
