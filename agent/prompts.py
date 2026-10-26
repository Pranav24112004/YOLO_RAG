"""
agent.prompts – Centralized prompt templates
=============================================
All LLM prompts live here so they can be reviewed, versioned,
and tweaked in one place without touching node logic.
"""

# ─── Intent Classification ───────────────────────────────────────────────

CLASSIFY_INTENT = """
You are a router for a YOLO (You Only Look Once) object detection research assistant.
Classify the user's message into exactly ONE of these categories:

1. GREETING  – casual greetings, hellos, how are you, thanks, bye, etc.
2. YOLO      – anything related to YOLO, object detection, computer vision, 
               deep learning architectures, neural networks, mAP, bounding boxes,
               anchor boxes, or any specific YOLO version (YOLOv1-v10, YOLOX, etc.)
               IMPORTANT: If the conversation history is about YOLO and the user asks
               a follow-up question (e.g. "tell me more", "what about its architecture",
               "how does it compare"), classify it as YOLO even if the message alone
               seems vague or off-topic.
3. OFF_TOPIC – anything NOT related to YOLO or object detection 
               (e.g. cooking recipes, weather, politics, general coding questions)
               Only classify as OFF_TOPIC if BOTH the message AND the conversation
               history have nothing to do with YOLO.

Recent Conversation History:
{chat_history}

User message: "{question}"

Respond with ONLY one word: GREETING, YOLO, or OFF_TOPIC
""".strip()

GREETING_RESPONSE = """
You are a friendly YOLO research assistant. The user has sent a greeting or casual message.
Respond warmly and briefly, then mention that you're here to help with anything about 
YOLO object detection papers and research.

User message: "{question}"

Keep the response short, friendly, and professional (2-3 sentences max).
""".strip()

OFF_TOPIC_RESPONSE = (
    "That's an interesting question, but I'm specifically designed to help with "
    "**YOLO (You Only Look Once)** object detection research. 🎯\n\n"
    "How can I help you understand YOLO? For example, you can ask me about:\n"
    "- Differences between YOLO versions (v1 through v10)\n"
    "- Architecture details, training strategies, or performance comparisons\n"
    "- Concepts like anchor boxes, mAP, or real-time detection"
)

# ─── Retriever ────────────────────────────────────────────────────────────

CONTEXTUALIZE_QUESTION = """
Chat History:
{chat_history}

Latest User Question: {question}

Refined Question:
Given the above history, rewrite the user's latest question to be standalone.
Explicitly replace ambiguous terms like "it", "that", "the previous model" with the actual subjects from history.
If the question is already standalone, return it exactly as is.
Do NOT answer the question.
Returns ONLY the refined question string.
""".strip()

# ─── Grader ───────────────────────────────────────────────────────────────

GRADE_DOCUMENTS = """
You are a strict professor grading a student's research.
User Question: {question}
Retrieved Documents: {documents}

Are these documents relevant to the question? 
If they are completely useless, reply "NO". 
If they contain at least some helpful info, reply "YES".
Only reply YES or NO strictly.
""".strip()

REWRITE_QUERY = """
The user asked: "{question}"
The search results were not relevant.
Please rewrite the query to be more specific or use better keywords to find relevant information in a vector database.
Return ONLY the new query, nothing else.
""".strip()

# ─── Generator ────────────────────────────────────────────────────────────

GENERATE_ANSWER = """
You are an expert on YOLO object detection papers.
Answer the user's question strictly based on the context provided below.

IMPORTANT CITATION RULES:
- After each key fact or claim, cite the source file and page in brackets, e.g. [YOLOv1.pdf, p.3]
- At the very end of your answer, add a "## Sources" section listing all cited papers.
- If the context doesn't contain the answer, say "I don't know based on the provided papers." and you will stop answering.

Context:
{context}

Chat History:
{history}

User Question: {question}
""".strip()
