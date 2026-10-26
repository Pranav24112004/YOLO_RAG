import streamlit as st
import requests
import uuid
import base64

# --- CONFIG ---
API_BASE_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="YOLO RAG Agent", layout="wide")

# --- STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_list" not in st.session_state:
    st.session_state.session_list = []

# --- HELPER FUNCTIONS ---
def fetch_sessions():
    """Fetch sessions from API and cache in state to avoid lag."""
    try:
        res = requests.get(f"{API_BASE_URL}/sessions", timeout=15)
        if res.status_code == 200:
            st.session_state.session_list = res.json()
    except Exception as e:
        print(f"Error fetching sessions: {e}")

def load_session(sid):
    if not sid:
        return
    st.session_state.session_id = sid
    try:
        hist_res = requests.get(f"{API_BASE_URL}/chat/{sid}", timeout=30)
        if hist_res.status_code == 200:
            data = hist_res.json()
            history = data.get("chat_history", [])
            normalized_history = []
            for msg in history:
                role = "assistant" if msg["role"] == "ai" else "user"
                normalized_history.append({"role": role, "content": msg["content"]})
            st.session_state.messages = normalized_history
        else:
            st.error(f"Failed to load history")
    except Exception as e:
        st.error(f"Error loading history: {e}")
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🗂️ Sessions")

    # Initial fetch if empty
    if not st.session_state.session_list:
        fetch_sessions()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("+ New", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            fetch_sessions()
            st.rerun()

    st.divider()

    # Session ID search/load
    search_id = st.text_input("🔍 Search Session ID", placeholder="Search by ID...")
    
    # Display Sessions from Cache
    sessions = st.session_state.session_list
    if sessions:
        # Filter locally if search text is provided
        filtered = [s for s in sessions if search_id.lower() in s['id'].lower()] if search_id else sessions
        
        if not filtered and search_id:
            st.caption("No matching sessions.")
        
        for s in filtered:
            label = f"💬 {s['id'][:12]}..."
            if st.session_state.session_id == s['id']:
                label = f"▶️ {s['id'][:12]}..."
                
            if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                load_session(s['id'])
    else:
        st.info("No sessions found. Try refreshing or start a new chat.")

    # Show current session ID for easy copy
    if st.session_state.session_id:
        st.divider()
        st.caption("Current Session ID:")
        st.code(st.session_state.session_id)

# --- MAIN CHAT ---
st.title("🤖 YOLO RAG Agent")

if st.session_state.session_id:
    st.caption(f"Session ID: {st.session_state.session_id}")
else:
    st.caption("New Session")

# --- CHAT UI ---
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source details for assistant messages — SLIDABLE CAROUSEL
        if msg["role"] == "assistant" and msg.get("source_details"):
            sources = msg["source_details"]
            num_sources = len(sources)

            slide_key = f"source_slide_{msg_idx}"
            if slide_key not in st.session_state:
                st.session_state[slide_key] = 0

            with st.expander("📄 View Sources", expanded=False):
                col_prev, col_info, col_next = st.columns([1, 3, 1])

                with col_prev:
                    if st.button("◀ Prev", key=f"prev_{msg_idx}", disabled=(st.session_state[slide_key] == 0)):
                        st.session_state[slide_key] -= 1
                        st.rerun()

                with col_info:
                    st.markdown(
                        f"<div style='text-align:center; padding-top:6px;'>"
                        f"<b>Source {st.session_state[slide_key] + 1} of {num_sources}</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col_next:
                    if st.button("Next ▶", key=f"next_{msg_idx}", disabled=(st.session_state[slide_key] >= num_sources - 1)):
                        st.session_state[slide_key] += 1
                        st.rerun()

                st.divider()

                current_idx = st.session_state[slide_key]
                src = sources[current_idx]

                st.markdown(f"**[{current_idx + 1}] {src['file']}** (Page {src['page']}) — _{src['section']}_")
                if src.get("snippet"):
                    st.caption(f'"{src["snippet"]}"')

                # Get the SAS link on demand
                link_cache_key = f"link_{msg_idx}_{current_idx}"
                if link_cache_key not in st.session_state:
                    st.session_state[link_cache_key] = None

                if not st.session_state[link_cache_key]:
                    if st.button("🔗 Generate PDF Link", key=f"gen_link_{msg_idx}_{current_idx}"):
                        with st.spinner("Generating secure link..."):
                            try:
                                params = {
                                    "file": src["file"],
                                    "page": src.get("page", 1),
                                }
                                res = requests.get(f"{API_BASE_URL}/source-link", params=params, timeout=30)
                                if res.status_code == 200:
                                    st.session_state[link_cache_key] = res.json().get("sas_url")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate link")
                            except Exception as e:
                                st.error(f"Error: {e}")

                if st.session_state[link_cache_key]:
                    st.success("Link generated! (Valid for 5 mins)")
                    st.link_button(
                        f"📄 Open {src['file']} at Page {src['page']}",
                        st.session_state[link_cache_key],
                        use_container_width=True
                    )
                    if st.button("🔄 Regenerate Link", key=f"re_gen_{msg_idx}_{current_idx}"):
                        st.session_state[link_cache_key] = None
                        st.rerun()

if prompt := st.chat_input("Ask something about YOLO..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"message": prompt}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id

                response = requests.post(f"{API_BASE_URL}/chat", json=payload)

                if response.status_code == 200:
                    data = response.json()

                    if not st.session_state.session_id:
                        st.session_state.session_id = data["session_id"]
                        fetch_sessions() # Refresh cache for new session

                    ai_response = data["response"]
                    source_details = data.get("source_details", [])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                        "source_details": source_details
                    })

                    st.rerun()

                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
