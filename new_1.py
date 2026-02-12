# ── Enhanced version of new.py ──
# Page-by-page extraction: text + image descriptions interleaved at correct positions
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
import requests

os.environ["TIKTOKEN_CACHE_DIR"] = r"C:\Users\GenAIBLRTKCUSR2\AiFriday\.cache\tiktoken"

import shutil
import streamlit as st
import base64
import fitz  # PyMuPDF – text extraction, image extraction, page rendering
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
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

# Vision-capable LLM – describes page screenshots (captures charts, diagrams, images)
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
#  Page-by-page extraction: text + vision
#  Text and visual descriptions are interleaved
#  at their correct positions per page
# ──────────────────────────────────────────────
def extract_pages_with_vision(pdf_path: str, dpi: int = 150) -> list[dict]:
    """
    For every page in the PDF:
      1. Extract text using PyMuPDF (preserves position within the page)
      2. Check if page has embedded images / visual content
      3. If yes → render the full page as a PNG screenshot → send to Vision LLM
      4. Return combined text + visual description per page

    Returns:
        List of dicts, one per page:
        [{"page": 1, "text": "...", "visual_description": "...",
          "has_visuals": bool, "num_images": int, "combined": "..."}, ...]
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_data: list[dict] = []

    progress = st.progress(0, text="Processing pages...")

    for page_num in range(total_pages):
        page = doc[page_num]

        # ── 1. Extract text from this page ──
        page_text = page.get_text("text").strip()

        # ── 2. Check for visual content on this page ──
        embedded_images = page.get_images(full=True)
        # Consider a page "visual" if it has embedded images
        # OR has very little text (likely a diagram/chart-heavy page)
        has_visual_content = len(embedded_images) > 0 or len(page_text) < 100

        visual_description = ""

        if has_visual_content:
            progress.progress(
                (page_num + 1) / total_pages,
                text=f"Page {page_num + 1}/{total_pages} — "
                     f"{len(embedded_images)} image(s) found, describing with Vision LLM..."
            )

            # ── 3. Render entire page as a PNG screenshot ──
            # This captures EVERYTHING: text, images, charts, tables, vector graphics
            pix = page.get_pixmap(dpi=dpi)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64}"

            # ── 4. Send page screenshot to Vision LLM ──
            try:
                response = vision_llm.invoke([
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"This is page {page_num + 1} of a PDF document. "
                                    "Describe ALL visual elements on this page: images, charts, "
                                    "graphs, tables, diagrams, logos, icons, flowcharts, etc. "
                                    "For charts/graphs: describe the data, axes, labels, and trends. "
                                    "For tables: describe the rows, columns, and key data points. "
                                    "For diagrams/flowcharts: describe the structure and connections. "
                                    "If there are no meaningful visual elements (just plain text), "
                                    "reply with exactly: NO_VISUALS"
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                        ],
                    }
                ])
                desc = response.content if isinstance(response.content, str) else str(response.content)

                # If Vision LLM says no visuals, skip
                if "NO_VISUALS" in desc.upper():
                    visual_description = ""
                else:
                    visual_description = desc
            except Exception as e:
                visual_description = f"(Vision LLM error on page {page_num + 1}: {e})"
        else:
            progress.progress(
                (page_num + 1) / total_pages,
                text=f"Page {page_num + 1}/{total_pages} — text only, no images."
            )

        # ── 5. Combine text + visual description for this page ──
        combined = f"=== PAGE {page_num + 1} ===\n"
        if page_text:
            combined += page_text + "\n"
        if visual_description:
            combined += (
                f"\n[Visual content on page {page_num + 1}]: "
                f"{visual_description}\n"
            )
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
st.set_page_config(page_title="RAG PDF Summarizer (Text + Images)")
st.title("RAG-powered PDF Summarizer (Text + Images)")

upload_file = st.file_uploader("Upload a PDF", type="pdf")
if upload_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name    # ── Only extract/index ONCE per uploaded file (cache in session_state) ──
    file_id = upload_file.name + "_" + str(upload_file.size)

    if st.session_state.get("indexed_file_id") != file_id:
        # ── Step 1: Page-by-page extraction (text + vision interleaved) ──
        with st.spinner("Extracting text and describing visuals page by page..."):
            pages_data = extract_pages_with_vision(temp_file_path)

        st.session_state["pages_data"] = pages_data

        # ── Step 2: Chunking the combined (text + vision) content ──
        full_document = "\n\n".join(p["combined"] for p in pages_data)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n=== END PAGE", "\n=== PAGE", "\n\n", "\n", " "],
        )
        chunks = text_splitter.split_text(full_document)
        st.session_state["num_chunks"] = len(chunks)

        # ── Step 3: Embed and store in Chroma ──
        with st.spinner("Indexing document (text + images) ..."):
            vectordb = Chroma.from_texts(
                chunks, embedding_model, persist_directory="./chroma_index"
            )

        # ── Step 4: Retriever ──
        retriever = vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )
        st.session_state["retriever"] = retriever

        # Mark this file as indexed so we don't redo it on rerun
        st.session_state["indexed_file_id"] = file_id
        # Clear chat history for the new document
        st.session_state["messages"] = []
        st.session_state["chat_input_key"] = 0

        st.success("✅ Document indexed successfully!")
    else:
        # Reuse cached data from session_state
        pages_data = st.session_state["pages_data"]
        retriever = st.session_state["retriever"]

    # Show extraction summary (always visible)
    pages_with_visuals = [p for p in pages_data if p["has_visuals"]]
    total_images = sum(p["num_images"] for p in pages_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", len(pages_data))
    col2.metric("Pages with Visuals", len(pages_with_visuals))
    col3.metric("Embedded Images", total_images)

    # Show per-page details
    with st.expander("View page-by-page extraction details"):
        for p in pages_data:
            icon = "🖼️" if p["has_visuals"] else "📄"
            st.markdown(
                f"**{icon} Page {p['page']}** — "
                f"{len(p['text'])} chars text, "
                f"{p['num_images']} images, "
                f"vision: {'✅' if p['has_visuals'] else '—'}"
            )
            if p["has_visuals"]:
                preview = p["visual_description"][:300]
                if len(p["visual_description"]) > 300:
                    preview += "..."
                st.caption(preview)

    st.caption(
        f"📦 {st.session_state.get('num_chunks', '?')} chunks indexed "
        f"(text + visual descriptions interleaved at correct page positions)"
    )

    # ──────────────────────────────────────────────
    #  LangGraph RAG Workflow
    # ──────────────────────────────────────────────

    class GraphState(TypedDict):
        question: str
        history: str
        context: Optional[str]
        answer: Optional[str]

    def retrieve_node(state: GraphState) -> dict:
        """Retrieve top-k chunks (text + interleaved image descriptions) from Chroma."""
        docs = retriever.invoke(state["question"])
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    def generate_node(state: GraphState) -> dict:
        """Generate answer using LLM + retrieved context + chat history."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that answers questions about a PDF document. "
             "The context below contains text from the document AND descriptions of visual "
             "elements (charts, diagrams, tables, images) found on each page. "
             "Use ALL available context — both text and visual descriptions — to answer.\n\n"
             "If you cannot find the answer in the context, say so.\n\n"
             "CONTEXT:\n{context}"),
            ("human",
             "Conversation so far:\n{history}\n\n"
             "Current question: {question}\n\n"
             "Answer the current question in a helpful, concise way."),
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": state.get("context", ""),
            "history": state.get("history", ""),
            "question": state["question"],
        })
        return {"answer": answer}

    # --- Build the LangGraph ---
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    rag_graph = workflow.compile()

    # ──────────────────────────────────────────────
    #  Visualise the LangGraph workflow
    # ──────────────────────────────────────────────
    st.subheader("LangGraph Workflow")
    tab_mermaid, tab_ascii = st.tabs(["📊 Mermaid Diagram", "📝 ASCII"])

    with tab_mermaid:
        mermaid_code = rag_graph.get_graph().draw_mermaid()
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

    with tab_ascii:
        ascii_art = rag_graph.get_graph().draw_ascii()
        st.code(ascii_art, language=None)

    # Helper to run the graph
    def run_rag_graph(question: str, history: str = "") -> str:
        result = rag_graph.invoke({
            "question": question,
            "history": history,
            "context": None,
            "answer": None,
        })
        return result.get("answer", "I could not generate an answer.")

    # ── One-click summary button ──
    if st.button("Generate overall summary"):
        with st.spinner("Generating summary..."):
            summary = run_rag_graph(
                "Provide a concise high-level summary of this entire document, "
                "including any information from charts, diagrams or images."
            )
        st.subheader("Document Summary")
        st.write(summary)

    # ──────────────────────────────────────────────
    #  Chat UI with conversation context
    # ──────────────────────────────────────────────
    st.subheader("Chat with your document (LangGraph RAG)")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "chat_input_key" not in st.session_state:
        st.session_state["chat_input_key"] = 0

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

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

        with st.spinner("LangGraph RAG thinking..."):
            agent_answer = run_rag_graph(user_query, history_text)

        st.session_state["messages"].append({"role": "assistant", "content": agent_answer})

        st.session_state["chat_input_key"] += 1
        st.rerun()
