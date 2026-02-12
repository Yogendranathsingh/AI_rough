# ═══════════════════════════════════════════════════════════════
#  Advanced LangGraph RAG — PDF Summarizer (Text + Images)
#
#  Graph nodes:
#    1. rewrite_query  – reformulates user question for better retrieval
#    2. retrieve       – fetches top-k chunks from Chroma
#    3. grade_context  – LLM judges if retrieved context is relevant
#    4. generate       – produces the answer from context
#    5. halluc_check   – LLM verifies answer is grounded in context
#    6. fallback       – graceful "not found" response
#
#  Conditional edges:
#    grade_context → generate   (if relevant)
#    grade_context → fallback   (if not relevant)
#    halluc_check  → END        (if grounded)
#    halluc_check  → retrieve   (retry once if hallucinated)
# ═══════════════════════════════════════════════════════════════
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["TIKTOKEN_CACHE_DIR"] = r"C:\Users\GenAIBLRTKCUSR2\AiFriday\.cache\tiktoken"

import streamlit as st
import base64
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import tempfile
import httpx

client = httpx.Client(verify=False)

# ──────────────────────────────────────────────
#  LLM and Embedding setup
# ──────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-sWqh4w3oqqFiMJy9Om_qzg",
    http_client=client,
)

vision_llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="genailab-maas-gpt-4o",
    api_key="sk-sWqh4w3oqqFiMJy9Om_qzg",
    http_client=client,
    max_tokens=1024,
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-sWqh4w3oqqFiMJy9Om_qzg",
    http_client=client,
)


# ──────────────────────────────────────────────
#  Page-by-page extraction (same as new_1.py)
# ──────────────────────────────────────────────
def extract_pages_with_vision(pdf_path: str, dpi: int = 150) -> list[dict]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_data: list[dict] = []
    progress = st.progress(0, text="Processing pages...")

    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text("text").strip()
        embedded_images = page.get_images(full=True)
        has_visual_content = len(embedded_images) > 0 or len(page_text) < 100
        visual_description = ""

        if has_visual_content:
            progress.progress(
                (page_num + 1) / total_pages,
                text=f"Page {page_num + 1}/{total_pages} — describing visuals with Vision LLM..."
            )
            pix = page.get_pixmap(dpi=dpi)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"
            try:
                response = vision_llm.invoke([{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            f"This is page {page_num + 1} of a PDF. "
                            "Describe ALL visual elements: images, charts, graphs, tables, "
                            "diagrams, flowcharts. For charts describe data, axes, trends. "
                            "For tables describe rows, columns, key data. "
                            "If no visual elements, reply: NO_VISUALS"
                        )},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }])
                desc = response.content if isinstance(response.content, str) else str(response.content)
                if "NO_VISUALS" not in desc.upper():
                    visual_description = desc
            except Exception as e:
                visual_description = f"(Vision LLM error on page {page_num + 1}: {e})"
        else:
            progress.progress(
                (page_num + 1) / total_pages,
                text=f"Page {page_num + 1}/{total_pages} — text only."
            )

        combined = f"=== PAGE {page_num + 1} ===\n"
        if page_text:
            combined += page_text + "\n"
        if visual_description:
            combined += f"\n[Visual content on page {page_num + 1}]: {visual_description}\n"
        combined += f"=== END PAGE {page_num + 1} ===\n"

        pages_data.append({
            "page": page_num + 1,
            "text": page_text,
            "visual_description": visual_description,
            "has_visuals": bool(visual_description),
            "num_images": len(embedded_images),
            "combined": combined,
        })

    doc.close()
    progress.empty()
    return pages_data


# ──────────────────────────────────────────────
#  Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="Advanced LangGraph RAG", layout="wide")
st.title("🧠 Advanced LangGraph RAG — PDF Summarizer")
st.caption("6-node graph: rewrite → retrieve → grade → generate → hallucination check → answer")

upload_file = st.file_uploader("Upload a PDF", type="pdf")
if upload_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    file_id = upload_file.name + "_" + str(upload_file.size)

    if st.session_state.get("indexed_file_id") != file_id:
        with st.spinner("Extracting text and describing visuals page by page..."):
            pages_data = extract_pages_with_vision(temp_file_path)
        st.session_state["pages_data"] = pages_data

        full_document = "\n\n".join(p["combined"] for p in pages_data)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n=== END PAGE", "\n=== PAGE", "\n\n", "\n", " "],
        )
        chunks = text_splitter.split_text(full_document)
        st.session_state["num_chunks"] = len(chunks)

        with st.spinner("Indexing document (text + images)..."):
            vectordb = Chroma.from_texts(
                chunks, embedding_model, persist_directory="./chroma_index_v2"
            )
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        st.session_state["retriever"] = retriever
        st.session_state["indexed_file_id"] = file_id
        st.session_state["messages"] = []
        st.session_state["chat_input_key"] = 0
        st.success("✅ Document indexed successfully!")
    else:
        pages_data = st.session_state["pages_data"]
        retriever = st.session_state["retriever"]

    # ── Extraction summary ──
    pages_with_visuals = [p for p in pages_data if p["has_visuals"]]
    total_images = sum(p["num_images"] for p in pages_data)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages", len(pages_data))
    col2.metric("Pages with Visuals", len(pages_with_visuals))
    col3.metric("Embedded Images", total_images)
    col4.metric("Chunks Indexed", st.session_state.get("num_chunks", "?"))

    with st.expander("View page-by-page extraction details"):
        for p in pages_data:
            icon = "🖼️" if p["has_visuals"] else "📄"
            st.markdown(
                f"**{icon} Page {p['page']}** — "
                f"{len(p['text'])} chars, {p['num_images']} images, "
                f"vision: {'✅' if p['has_visuals'] else '—'}"
            )
            if p["has_visuals"]:
                preview = p["visual_description"][:300]
                if len(p["visual_description"]) > 300:
                    preview += "..."
                st.caption(preview)

    # ══════════════════════════════════════════════
    #  ADVANCED LANGGRAPH — 6 nodes, conditional edges
    # ══════════════════════════════════════════════

    class GraphState(TypedDict):
        question: str           # original user question
        rewritten_query: str    # improved query for retrieval
        history: str            # conversation history
        context: Optional[str]  # retrieved chunks
        answer: Optional[str]   # generated answer
        is_relevant: Optional[bool]   # did context pass grading?
        is_grounded: Optional[bool]   # did answer pass hallucination check?
        retries: int            # how many times we've retried retrieval
        node_trace: str         # trace of which nodes ran (for UI)

    # ── Node 1: REWRITE QUERY ──
    def rewrite_query_node(state: GraphState) -> dict:
        """Reformulate the user question for better vector search retrieval."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a query rewriter. Given a user question and conversation history, "
             "rewrite the question to be more specific and self-contained for vector search. "
             "Output ONLY the rewritten query, nothing else."),
            ("human",
             "Conversation history:\n{history}\n\n"
             "Original question: {question}\n\n"
             "Rewritten query:"),
        ])
        chain = prompt | llm | StrOutputParser()
        rewritten = chain.invoke({
            "history": state.get("history", ""),
            "question": state["question"],
        })
        return {
            "rewritten_query": rewritten.strip(),
            "node_trace": state.get("node_trace", "") + "✏️ rewrite → ",
        }

    # ── Node 2: RETRIEVE ──
    def retrieve_node(state: GraphState) -> dict:
        """Retrieve top-k chunks using the rewritten query."""
        query = state.get("rewritten_query") or state["question"]
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return {
            "context": context,
            "node_trace": state.get("node_trace", "") + "🔍 retrieve → ",
        }

    # ── Node 3: GRADE CONTEXT ──
    def grade_context_node(state: GraphState) -> dict:
        """LLM judges whether retrieved context is relevant to the question."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a relevance grader. Given a question and retrieved context, "
             "decide if the context contains information relevant to answering the question. "
             "Reply with exactly one word: RELEVANT or IRRELEVANT"),
            ("human",
             "Question: {question}\n\n"
             "Retrieved context:\n{context}\n\n"
             "Verdict:"),
        ])
        chain = prompt | llm | StrOutputParser()
        verdict = chain.invoke({
            "question": state["question"],
            "context": state.get("context", ""),
        })
        is_relevant = "RELEVANT" in verdict.upper() and "IRRELEVANT" not in verdict.upper()
        return {
            "is_relevant": is_relevant,
            "node_trace": state.get("node_trace", "") + f"📋 grade({'✅' if is_relevant else '❌'}) → ",
        }

    # ── Node 4: GENERATE ──
    def generate_node(state: GraphState) -> dict:
        """Generate the answer using context + history."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering questions about a PDF document. "
             "The context includes text AND descriptions of visual elements (charts, "
             "diagrams, tables, images). Use ALL context to answer.\n\n"
             "IMPORTANT: Only use information from the provided context. "
             "If the context doesn't contain the answer, say so.\n\n"
             "CONTEXT:\n{context}"),
            ("human",
             "Conversation so far:\n{history}\n\n"
             "Question: {question}\n\n"
             "Answer:"),
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": state.get("context", ""),
            "history": state.get("history", ""),
            "question": state["question"],
        })
        return {
            "answer": answer,
            "node_trace": state.get("node_trace", "") + "💡 generate → ",
        }

    # ── Node 5: HALLUCINATION CHECK ──
    def halluc_check_node(state: GraphState) -> dict:
        """LLM checks if the generated answer is grounded in the retrieved context."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a hallucination detector. Given a context and an answer, "
             "determine if the answer is fully supported by the context. "
             "Reply with exactly one word: GROUNDED or HALLUCINATED"),
            ("human",
             "Context:\n{context}\n\n"
             "Answer:\n{answer}\n\n"
             "Verdict:"),
        ])
        chain = prompt | llm | StrOutputParser()
        verdict = chain.invoke({
            "context": state.get("context", ""),
            "answer": state.get("answer", ""),
        })
        is_grounded = "GROUNDED" in verdict.upper() and "HALLUCINATED" not in verdict.upper()
        retries = state.get("retries", 0)
        return {
            "is_grounded": is_grounded,
            "retries": retries + (0 if is_grounded else 1),
            "node_trace": state.get("node_trace", "") + f"🛡️ halluc({'✅' if is_grounded else '🔄'}) → ",
        }

    # ── Node 6: FALLBACK ──
    def fallback_node(state: GraphState) -> dict:
        """Produce a graceful 'not found' response."""
        return {
            "answer": (
                "I couldn't find relevant information in the document to answer "
                "your question. Could you try rephrasing, or ask about a different "
                "topic covered in the PDF?"
            ),
            "node_trace": state.get("node_trace", "") + "⚠️ fallback → ",
        }

    # ── Conditional edge functions ──
    def route_after_grading(state: GraphState) -> str:
        """After grading: go to generate if relevant, else fallback."""
        if state.get("is_relevant"):
            return "generate"
        return "fallback"

    def route_after_halluc_check(state: GraphState) -> str:
        """After hallucination check: END if grounded, retry retrieve if not (max 1 retry)."""
        if state.get("is_grounded"):
            return END
        # Allow at most 1 retry to avoid infinite loops
        if state.get("retries", 0) >= 2:
            return END
        return "retrieve"

    # ── Build the graph ──
    workflow = StateGraph(GraphState)

    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("halluc_check", halluc_check_node)
    workflow.add_node("fallback", fallback_node)

    # Edges
    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade_context")

    # Conditional: grade → generate or fallback
    workflow.add_conditional_edges("grade_context", route_after_grading, {
        "generate": "generate",
        "fallback": "fallback",
    })

    workflow.add_edge("generate", "halluc_check")

    # Conditional: halluc_check → END or retry retrieve
    workflow.add_conditional_edges("halluc_check", route_after_halluc_check, {
        END: END,
        "retrieve": "retrieve",
    })

    workflow.add_edge("fallback", END)

    rag_graph = workflow.compile()

    # ══════════════════════════════════════════════
    #  Visualise the graph
    # ══════════════════════════════════════════════
    st.subheader("🗺️ LangGraph Workflow (Advanced)")
    tab_mermaid, tab_ascii = st.tabs(["📊 Mermaid Diagram", "📝 ASCII"])

    with tab_mermaid:
        mermaid_code = rag_graph.get_graph().draw_mermaid()
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

    with tab_ascii:
        ascii_art = rag_graph.get_graph().draw_ascii()
        st.code(ascii_art, language=None)

    # ── Helper to run the graph ──
    def run_rag_graph(question: str, history: str = "") -> dict:
        """Run the full graph and return the final state (answer + trace)."""
        result = rag_graph.invoke({
            "question": question,
            "rewritten_query": "",
            "history": history,
            "context": None,
            "answer": None,
            "is_relevant": None,
            "is_grounded": None,
            "retries": 0,
            "node_trace": "",
        })
        return result

    # ── One-click summary ──
    if st.button("📝 Generate overall summary"):
        with st.spinner("Running advanced RAG graph for summary..."):
            result = run_rag_graph(
                "Provide a concise high-level summary of this entire document, "
                "including any information from charts, diagrams, tables or images."
            )
        st.subheader("Document Summary")
        st.write(result.get("answer", "Could not generate summary."))
        with st.expander("🔍 Graph execution trace"):
            st.code(result.get("node_trace", ""))

    # ══════════════════════════════════════════════
    #  Chat UI
    # ══════════════════════════════════════════════
    st.subheader("💬 Chat with your document (Advanced LangGraph RAG)")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_input_key" not in st.session_state:
        st.session_state["chat_input_key"] = 0

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**🤖 Assistant:** {msg['content']}")
            if msg.get("trace"):
                with st.expander("🔍 Graph trace for this answer", expanded=False):
                    st.code(msg["trace"])

    user_query = st.text_input(
        "Type your message and press Enter:",
        key=f"chat_input_{st.session_state['chat_input_key']}",
    )

    if user_query and (
        not st.session_state["messages"]
        or st.session_state["messages"][-1]["role"] != "user"
        or st.session_state["messages"][-1]["content"] != user_query
    ):
        st.session_state["messages"].append({"role": "user", "content": user_query})

        history_text = ""
        for msg in st.session_state["messages"]:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{prefix}: {msg['content']}\n"

        with st.spinner("🧠 Advanced LangGraph RAG thinking..."):
            result = run_rag_graph(user_query, history_text)

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result.get("answer", "I could not generate an answer."),
            "trace": result.get("node_trace", ""),
        })

        st.session_state["chat_input_key"] += 1
        st.rerun()
