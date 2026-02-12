import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
import requests

os.environ["TIKTOKEN_CACHE_DIR"] = r"C:\Users\GenAIBLRTKCUSR2\AiFriday\.cache\tiktoken"
# os.environ["XDG_CACHE_HOME"] = r"C:\Users\GenAIBLRTKCUSR2\AiFriday\.cache"

import shutil
import streamlit as st 
from pdfminer.high_level import extract_text 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import tempfile 
import os 
import httpx 
client = httpx.Client(verify=False)   

# Always start with a fresh Chroma index for this session
# CHROMA_DIR = "./chroma_index"
# if os.path.exists(CHROMA_DIR):
#     shutil.rmtree(CHROMA_DIR)
 
# LLM and Embedding setup 
llm = ChatOpenAI( 
   base_url="https://genailab.tcs.in", 
   model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
   api_key="sk-sWqh4w3oqqFiMJy9Om_qzg", 
   http_client=client 
) 
embedding_model = OpenAIEmbeddings( 
   base_url="https://genailab.tcs.in", 
   model="azure/genailab-maas-text-embedding-3-large", 
   api_key= "sk-sWqh4w3oqqFiMJy9Om_qzg", 
   http_client=client
   ) 

st.set_page_config(page_title="RAG PDF Summarizer")



st.title("     RAG-powered PDF Summarizer")
upload_file = st.file_uploader("Upload a PDF", type="pdf")
if upload_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name
    # Step 1: Extract text
    raw_text = extract_text(temp_file_path)
    # Step 2: Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Step 3: Embed and store in Chroma
    with st.spinner("Indexing document..."):
        vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_index")
        # vectordb.persist()

    # Step 4: Retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # ──────────────────────────────────────────────
    #  LangGraph RAG Workflow
    # ──────────────────────────────────────────────

    # --- State schema for the graph ---
    class GraphState(TypedDict):
        question: str                       # current user question
        history: str                        # formatted conversation history
        context: Optional[str]              # retrieved document chunks
        answer: Optional[str]               # final LLM answer

    # --- Node 1: Retrieve relevant chunks ---
    def retrieve_node(state: GraphState) -> dict:
        """Retrieve top-k document chunks from Chroma based on the question."""
        docs = retriever.invoke(state["question"])
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    # --- Node 2: Generate answer using LLM + context + history ---
    def generate_node(state: GraphState) -> dict:
        """Use the LLM to answer the question given retrieved context and chat history."""
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant that answers questions about a PDF document. "
             "Use the following retrieved document context to answer. "
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

    # Flow: retrieve ➜ generate ➜ END
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph (creates a runnable)
    rag_graph = workflow.compile()

    # ──────────────────────────────────────────────
    #  Visualise the LangGraph workflow
    # ──────────────────────────────────────────────
    st.subheader("LangGraph Workflow")

    # Tab 1: interactive Mermaid diagram  |  Tab 2: ASCII art
    tab_mermaid, tab_ascii = st.tabs(["📊 Mermaid Diagram", "📝 ASCII"])

    with tab_mermaid:
        mermaid_code = rag_graph.get_graph().draw_mermaid()
        # Streamlit renders Mermaid inside a ```mermaid block
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

    # Optional: one-click summary button
    if st.button("Generate overall summary"):
        with st.spinner("Generating summary..."):
            summary = run_rag_graph(
                "Provide a concise high-level summary of this entire document."
            )
        st.subheader("Document Summary")
        st.write(summary)

    # ──────────────────────────────────────────────
    #  Chat UI with conversation context
    # ──────────────────────────────────────────────
    st.subheader("Chat with your document (LangGraph RAG)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Track which input key to use (increments after each answer to clear the box)
    if "chat_input_key" not in st.session_state:
        st.session_state["chat_input_key"] = 0

    # Display previous messages
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

    # Chat input (unique key each turn so the box resets to empty)
    user_query = st.text_input(
        "Type your message and press Enter:",
        key=f"chat_input_{st.session_state['chat_input_key']}",
    )

    # Only handle a NEW question once
    if user_query and (
        not st.session_state["messages"]
        or st.session_state["messages"][-1]["role"] != "user"
        or st.session_state["messages"][-1]["content"] != user_query
    ):
        # 1) Add user message to history
        st.session_state["messages"].append({"role": "user", "content": user_query})

        # 2) Build conversation history string
        history_text = ""
        for msg in st.session_state["messages"]:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{prefix}: {msg['content']}\n"

        # 3) Run the LangGraph RAG workflow
        with st.spinner("LangGraph RAG thinking..."):
            agent_answer = run_rag_graph(user_query, history_text)

        # 4) Add assistant answer to history
        st.session_state["messages"].append({"role": "assistant", "content": agent_answer})

        # 5) Increment the key so the input box is recreated empty
        st.session_state["chat_input_key"] += 1
        st.rerun()