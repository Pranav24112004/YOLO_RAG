"""
agent.nodes.retriever – Query contextualization & vector search
"""

from azure.search.documents.models import VectorizedQuery, QueryType

from agent.state import AgentState
from agent import prompts
import config.settings as settings
from core.logger import log_event


def retrieve_node(state: AgentState):
    """Contextualize the question, embed it, and search Azure Cognitive Search."""
    from agent.nodes import llm, search_client, embedding_client

    question = state["question"]

    # --- Contextualize using chat history ---
    if state.get("chat_history") and state.get("retry_count", 0) == 0:
        print("[Librarian]: Contextualizing query...")
        # Line 19 area in retriever.py
        print(f"[DEBUG] chat_history length: {len(state.get('chat_history', []))}")
        print(f"[DEBUG] retry_count: {state.get('retry_count', 0)}")
        prompt = prompts.CONTEXTUALIZE_QUESTION.format(
            chat_history=state["chat_history"],
            question=question,
        )
        refined_question = llm.invoke(prompt).content.strip()
        if refined_question != question:
            print(f"[Librarian]: Refined to '{refined_question}'")
            question = refined_question

    print(f"[Search] Searching '{question}'...")

    # --- Embed & search ---
    query_vector = embedding_client.embeddings.create(
        model=settings.EMBEDDING_MODEL, inputs=[question]
    ).data[0].embedding

    results = search_client.search(
        search_text=question,
        vector_queries=[
            VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="vector")
        ],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="my-semantic-config",
        top=5,
    )

    docs = []
    source_details = []
    seen_sources = set()

    for d in results:
        section  = d.get("parent_section", "General")
        page_num = d.get("source_page", "?")
        src_file = d.get("source_file", "Unknown")
        content  = d["content"]

        docs.append(
            f"[Source: {src_file} | Section: {section} | Page: {page_num}]\n"
            f"{content}"
        )

        source_key = f"{src_file}_p{page_num}"
        if source_key not in seen_sources:
            # Strip the [Section: ...] prefix if it exists in the raw content
            import re
            clean_content = re.sub(r"^\[Section:.*?\]", "", content, flags=re.DOTALL).strip()
            
            snippet = clean_content[:200].strip()
            if len(clean_content) > 200:
                snippet += "..."

            # No PDF rendering here — images are loaded lazily via /source-image
            source_details.append({
                "file":    src_file,
                "page":    page_num,
                "section": section,
                "snippet": snippet,
            })
            seen_sources.add(source_key)

    log_event(
        state.get("session_id"), 
        "RETRIEVAL", 
        {"search_query": question, "docs_found": len(docs)}
    )

    return {"documents": docs, "question": question, "source_details": source_details}
