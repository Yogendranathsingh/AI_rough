# ═══════════════════════════════════════════════════════════════
#  Advanced LangGraph RAG — PDF Summarizer (Text + Images)
#  WITH INPUT & OUTPUT GUARDRAILS
#
#  Graph nodes:
#    0. input_guardrail  – blocks injection, toxicity, PII redaction
#    1. rewrite_query    – reformulates user question for better retrieval
#    2. retrieve         – fetches top-k chunks from Chroma (over-fetches k=10)
#    3. rank_context     – LLM scores each chunk 0-10 for relevance, keeps top-5
#    4. grade_context    – LLM judges if ranked context is relevant overall
#    5. generate         – produces the answer from context
#    6. output_guardrail – blocks toxic output, redacts PII leakage
#    7. halluc_check     – LLM verifies answer is grounded in context
#    8. fallback         – graceful "not found" response
#
#  Conditional edges:
#    input_guardrail → rewrite_query (if safe) or END (if blocked)
#    grade_context   → generate   (if relevant)
#    grade_context   → fallback   (if not relevant)
#    halluc_check    → END        (if grounded)
#    halluc_check    → retrieve   (retry once if hallucinated)
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
import re

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
#  GUARDRAILS — Input & Output Safety
# ──────────────────────────────────────────────

# ── PII patterns ──
PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "Email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "Phone (US)": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "Phone (IN)": r"\b(?:\+91[-.\s]?)?\d{10}\b",
    "Aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN (India)": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "Passport": r"\b[A-Z]\d{7}\b",
    "IP Address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}

# ── Prompt injection / jailbreak keywords ──
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a|an)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"bypass\s+(safety|content|filter|guardrail)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"do\s+anything\s+now",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"reveal\s+(your|the)\s+(system|initial)\s+prompt",
    r"show\s+(me\s+)?(your|the)\s+instructions",
    r"what\s+are\s+your\s+(system\s+)?instructions",
]

# ── Toxic / harmful content keywords ──
TOXIC_PATTERNS = [
    r"\b(kill|murder|assassinate|bomb|terror|weapon|exploit|hack)\b",
    r"\b(suicide|self[- ]?harm)\b",
    r"\b(racist|sexist|homophob|slur)\b",
    r"\b(illegal\s+drug|make\s+a\s+bomb|how\s+to\s+hack)\b",
]

# ── Off-topic patterns (not about the document) ──
OFF_TOPIC_PATTERNS = [
    r"\b(write\s+me\s+(a|an)\s+(poem|essay|story|code|script|song))\b",
    r"\b(translate\s+.+\s+to\s+)",
    r"\b(what\s+is\s+the\s+weather)\b",
    r"\b(tell\s+me\s+a\s+joke)\b",
    r"\b(who\s+is\s+the\s+president)\b",
    r"\b(play\s+(a\s+)?game)\b",
]


def detect_pii(text: str) -> list[dict]:
    """Detect PII patterns in text. Returns list of {type, match} dicts."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            findings.append({"type": pii_type, "match": m})
    return findings


def redact_pii(text: str) -> str:
    """Replace detected PII with [REDACTED] placeholders."""
    redacted = text
    for pii_type, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[{pii_type} REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted


def check_prompt_injection(text: str) -> tuple[bool, str]:
    """Check for prompt injection / jailbreak attempts.
    Returns (is_blocked, reason)."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True, f"Prompt injection detected (pattern: {pattern})"
    return False, ""


def check_toxic_content(text: str) -> tuple[bool, str]:
    """Check for toxic / harmful content.
    Returns (is_blocked, reason)."""
    text_lower = text.lower()
    for pattern in TOXIC_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return True, f"Potentially harmful content detected: '{match.group()}'"
    return False, ""


def check_off_topic(text: str) -> tuple[bool, str]:
    """Check if query is clearly off-topic (not about the document).
    Returns (is_off_topic, reason)."""
    text_lower = text.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return True, "Query appears to be off-topic and not related to the uploaded document."
    return False, ""


def input_guardrail_check(query: str) -> dict:
    """Run all input guardrails. Returns dict with 'passed', 'flags', 'sanitized_query'."""
    flags = []

    # 1. Empty / too short
    if not query or len(query.strip()) < 3:
        return {
            "passed": False,
            "flags": ["⚠️ Query is too short. Please ask a meaningful question."],
            "sanitized_query": query,
        }

    # 2. Too long (> 2000 chars)
    if len(query) > 2000:
        return {
            "passed": False,
            "flags": ["⚠️ Query is too long (>2000 characters). Please shorten your question."],
            "sanitized_query": query,
        }

    # 3. Prompt injection
    is_injection, reason = check_prompt_injection(query)
    if is_injection:
        return {
            "passed": False,
            "flags": [f"🚫 {reason}. This query has been blocked for safety."],
            "sanitized_query": query,
        }

    # 4. Toxic content
    is_toxic, reason = check_toxic_content(query)
    if is_toxic:
        return {
            "passed": False,
            "flags": [f"🚫 {reason}. Please rephrase your question."],
            "sanitized_query": query,
        }

    # 5. Off-topic check (soft warning, still allowed)
    is_off_topic, reason = check_off_topic(query)
    if is_off_topic:
        flags.append(f"⚠️ {reason} The answer may not be useful.")

    # 6. PII detection (warn + sanitize, still allowed)
    pii_found = detect_pii(query)
    if pii_found:
        pii_types = ", ".join(set(p["type"] for p in pii_found))
        flags.append(f"⚠️ PII detected in your query ({pii_types}). It has been redacted before processing.")
        sanitized = redact_pii(query)
    else:
        sanitized = query

    return {
        "passed": True,
        "flags": flags,
        "sanitized_query": sanitized,
    }


def output_guardrail_check(answer: str) -> dict:
    """Run all output guardrails. Returns dict with 'passed', 'flags', 'sanitized_answer'."""
    flags = []

    # 1. Toxic content in output
    is_toxic, reason = check_toxic_content(answer)
    if is_toxic:
        return {
            "passed": False,
            "flags": [f"🚫 Output blocked: {reason}"],
            "sanitized_answer": (
                "The generated answer was blocked by output safety guardrails. "
                "Please try rephrasing your question."
            ),
        }

    # 2. Prompt injection leakage (LLM echoing system prompt)
    is_injection, reason = check_prompt_injection(answer)
    if is_injection:
        return {
            "passed": False,
            "flags": ["🚫 Output blocked: potential system prompt leakage detected."],
            "sanitized_answer": (
                "The generated answer was blocked by output safety guardrails. "
                "Please try a different question."
            ),
        }

    # 3. PII leakage in output (redact)
    pii_found = detect_pii(answer)
    if pii_found:
        pii_types = ", ".join(set(p["type"] for p in pii_found))
        flags.append(f"⚠️ PII detected in output ({pii_types}) — redacted for safety.")
        sanitized = redact_pii(answer)
    else:
        sanitized = answer

    # 4. LLM uncertainty detection
    uncertainty_phrases = [
        "i'm not sure", "i am not sure", "i don't know", "i cannot determine",
        "it's unclear", "the context doesn't", "the context does not",
        "no information", "not mentioned", "cannot find",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in uncertainty_phrases):
        flags.append("ℹ️ The model expressed uncertainty. The answer may be incomplete.")

    return {
        "passed": True,
        "flags": flags,
        "sanitized_answer": sanitized,
    }


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
st.title("🧠 Advanced LangGraph RAG — PDF Summarizer (with Guardrails)")
st.caption("9-node graph: input guard → rewrite → retrieve → rank → grade → generate → output guard → hallucination check → answer")

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
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        st.session_state["retriever"] = retriever
        st.session_state["vectordb"] = vectordb
        st.session_state["indexed_file_id"] = file_id
        st.session_state["messages"] = []
        st.session_state["chat_input_key"] = 0
        st.success("✅ Document indexed successfully!")
    else:
        pages_data = st.session_state["pages_data"]
        retriever = st.session_state["retriever"]
        vectordb = st.session_state["vectordb"]

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
    #  ADVANCED LANGGRAPH — 9 nodes, conditional edges
    # ══════════════════════════════════════════════

    class GraphState(TypedDict):
        question: str           # original user question
        rewritten_query: str    # improved query for retrieval
        history: str            # conversation history
        context: Optional[str]  # retrieved chunks (raw from retriever)
        ranked_context: Optional[str]  # context after LLM ranking (top chunks only)
        ranking_scores: Optional[str]  # human-readable ranking results for trace
        answer: Optional[str]   # generated answer
        is_relevant: Optional[bool]   # did context pass grading?
        is_grounded: Optional[bool]   # did answer pass hallucination check?
        retries: int            # how many times we've retried retrieval
        node_trace: str         # trace of which nodes ran (for UI)
        input_blocked: bool     # did input guardrail block the query?
        input_flags: list       # warnings/errors from input guardrail
        output_flags: list      # warnings/errors from output guardrail
        skip_grading: bool      # if True, bypass grade_context and go straight to generate

    # ── Node 0: INPUT GUARDRAIL ──
    def input_guardrail_node(state: GraphState) -> dict:
        """Run input guardrails: injection detection, toxicity, PII, length checks."""
        result = input_guardrail_check(state["question"])
        if not result["passed"]:
            return {
                "input_blocked": True,
                "input_flags": result["flags"],
                "answer": result["flags"][0],  # blocked message becomes the answer
                "node_trace": state.get("node_trace", "") + "🛑 input_guard(BLOCKED) → ",
            }
        return {
            "input_blocked": False,
            "input_flags": result["flags"],
            "question": result["sanitized_query"],  # use sanitized (PII-redacted) query
            "node_trace": state.get("node_trace", "") + "🛡️ input_guard(✅) → ",
        }

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
        """Retrieve top-k chunks using the rewritten query.
        Uses k=15 for summary queries (skip_grading=True) to get broader coverage."""
        query = state.get("rewritten_query") or state["question"]
        if state.get("skip_grading"):
            # For summary: use a larger k to cover more of the document
            summary_retriever = vectordb.as_retriever(
                search_type="similarity", search_kwargs={"k": 15}
            )
            docs = summary_retriever.invoke(query)
        else:
            docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return {
            "context": context,
            "node_trace": state.get("node_trace", "") + f"🔍 retrieve(k={len(docs)}) → ",
        }

    # ── Node 3: RANK CONTEXT (LLM-based cross-encoder style reranking) ──
    def rank_context_node(state: GraphState) -> dict:
        """LLM scores each retrieved chunk for relevance (0-10), keeps top 5.
        Skipped when skip_grading=True (summary queries use all retrieved chunks)."""
        if state.get("skip_grading"):
            # For summary: use all retrieved chunks as-is, no ranking needed
            return {
                "ranked_context": state.get("context", ""),
                "ranking_scores": "Ranking skipped (summary mode)",
                "node_trace": state.get("node_trace", "") + "📊 rank(⏭️ skipped) → ",
            }

        raw_context = state.get("context", "")
        question = state["question"]

        # Split back into individual chunks
        chunks = [c.strip() for c in raw_context.split("\n\n---\n\n") if c.strip()]

        if not chunks:
            return {
                "ranked_context": "",
                "ranking_scores": "No chunks to rank",
                "node_trace": state.get("node_trace", "") + "📊 rank(0 chunks) → ",
            }

        # LLM scores each chunk for relevance to the question
        scoring_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a relevance scorer. Given a question and a text chunk from a document, "
             "rate how relevant the chunk is to answering the question.\n\n"
             "Reply with ONLY a single integer from 0 to 10, where:\n"
             "  0 = completely irrelevant\n"
             "  5 = somewhat relevant\n"
             " 10 = highly relevant and directly answers the question\n\n"
             "Output ONLY the number, nothing else."),
            ("human",
             "Question: {question}\n\n"
             "Text chunk:\n{chunk}\n\n"
             "Relevance score (0-10):"),
        ])
        scoring_chain = scoring_prompt | llm | StrOutputParser()

        scored_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                score_text = scoring_chain.invoke({
                    "question": question,
                    "chunk": chunk,
                })
                # Extract the numeric score
                score_digits = re.search(r"\d+", score_text.strip())
                score = int(score_digits.group()) if score_digits else 0
                score = max(0, min(10, score))  # clamp to 0-10
            except Exception:
                score = 5  # default mid-score on error
            scored_chunks.append((score, i, chunk))

        # Sort by score descending, keep top 5
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_n = 5
        top_chunks = scored_chunks[:top_n]

        # Build ranked context from top chunks
        ranked_context = "\n\n---\n\n".join(chunk for _, _, chunk in top_chunks)

        # Build human-readable ranking summary for trace
        ranking_lines = []
        for score, idx, chunk in scored_chunks:
            preview = chunk[:80].replace("\n", " ")
            marker = " ✅" if (score, idx, chunk) in top_chunks else " ❌"
            ranking_lines.append(f"  [{score:>2}/10]{marker} chunk {idx+1}: {preview}...")
        ranking_summary = f"Ranked {len(chunks)} chunks → kept top {len(top_chunks)}:\n" + "\n".join(ranking_lines)

        return {
            "ranked_context": ranked_context,
            "ranking_scores": ranking_summary,
            "node_trace": state.get("node_trace", "") + f"📊 rank({len(chunks)}→{len(top_chunks)}) → ",
        }

    # ── Node 4: GRADE CONTEXT ──
    def grade_context_node(state: GraphState) -> dict:
        """LLM judges whether retrieved context is relevant to the question.
        Skipped (auto-RELEVANT) when skip_grading=True (e.g., summary queries)."""
        if state.get("skip_grading"):
            return {
                "is_relevant": True,
                "node_trace": state.get("node_trace", "") + "📋 grade(⏭️ skipped) → ",
            }
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a relevance grader. Given a question and retrieved context, "
             "decide if the context contains information relevant to answering the question. "
             "Reply with exactly one word: RELEVANT or IRRELEVANT"),
            ("human",
             "Question: {question}\n\n"
             "Retrieved context:\n{context}\n\n"             "Verdict:"),
        ])
        chain = prompt | llm | StrOutputParser()
        verdict = chain.invoke({
            "question": state["question"],
            "context": state.get("ranked_context", "") or state.get("context", ""),
        })
        is_relevant = "RELEVANT" in verdict.upper() and "IRRELEVANT" not in verdict.upper()
        return {
            "is_relevant": is_relevant,
            "node_trace": state.get("node_trace", "") + f"📋 grade({'✅' if is_relevant else '❌'}) → ",
        }

    # ── Node 5: GENERATE ──
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
             "Conversation so far:\n{history}\n\n"             "Question: {question}\n\n"
             "Answer:"),
        ])
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": state.get("ranked_context", "") or state.get("context", ""),
            "history": state.get("history", ""),
            "question": state["question"],
        })
        return {
            "answer": answer,
            "node_trace": state.get("node_trace", "") + "💡 generate → ",
        }

    # ── Node 6: HALLUCINATION CHECK ──
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
            "context": state.get("ranked_context", "") or state.get("context", ""),
            "answer": state.get("answer", ""),
        })
        is_grounded = "GROUNDED" in verdict.upper() and "HALLUCINATED" not in verdict.upper()
        retries = state.get("retries", 0)
        return {
            "is_grounded": is_grounded,
            "retries": retries + (0 if is_grounded else 1),
            "node_trace": state.get("node_trace", "") + f"🛡️ halluc({'✅' if is_grounded else '🔄'}) → ",
        }

    # ── Node 7: FALLBACK ──
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

    # ── Node 8: OUTPUT GUARDRAIL ──
    def output_guardrail_node(state: GraphState) -> dict:
        """Run output guardrails: toxicity, PII leakage, injection leakage, uncertainty."""
        answer = state.get("answer", "")
        result = output_guardrail_check(answer)
        return {
            "answer": result["sanitized_answer"],
            "output_flags": result["flags"],
            "node_trace": state.get("node_trace", "") + (
                "🛡️ output_guard(✅) → " if result["passed"] else "🛑 output_guard(BLOCKED) → "
            ),
        }

    # ── Conditional edge functions ──
    def route_after_input_guardrail(state: GraphState) -> str:
        """After input guardrail: proceed to rewrite if safe, else skip to END."""
        if state.get("input_blocked"):
            return END
        return "rewrite_query"

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
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rank_context", rank_context_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("halluc_check", halluc_check_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("output_guardrail", output_guardrail_node)

    # Edges — start with input guardrail
    workflow.set_entry_point("input_guardrail")

    # Conditional: input_guardrail → rewrite_query (safe) or END (blocked)
    workflow.add_conditional_edges("input_guardrail", route_after_input_guardrail, {
        "rewrite_query": "rewrite_query",
        END: END,
    })

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "rank_context")
    workflow.add_edge("rank_context", "grade_context")

    # Conditional: grade → generate or fallback
    workflow.add_conditional_edges("grade_context", route_after_grading, {
        "generate": "generate",
        "fallback": "fallback",
    })

    # generate → output_guardrail (check answer safety before halluc check)
    workflow.add_edge("generate", "output_guardrail")
    workflow.add_edge("output_guardrail", "halluc_check")

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
    def run_rag_graph(question: str, history: str = "", skip_grading: bool = False) -> dict:
        """Run the full graph and return the final state (answer + trace)."""
        result = rag_graph.invoke({
            "question": question,
            "rewritten_query": "",
            "history": history,
            "context": None,
            "ranked_context": None,
            "ranking_scores": None,
            "answer": None,
            "is_relevant": None,
            "is_grounded": None,
            "retries": 0,
            "node_trace": "",
            "input_blocked": False,
            "input_flags": [],
            "output_flags": [],
            "skip_grading": skip_grading,
        })
        return result

    # ── One-click summary (skip grading — broad queries get full coverage) ──
    if st.button("📝 Generate overall summary"):
        with st.spinner("Running advanced RAG graph for summary..."):
            result = run_rag_graph(
                "Provide a concise high-level summary of this entire document, "
                "including any information from charts, diagrams, tables or images.",
                skip_grading=True,
            )
        st.subheader("Document Summary")
        answer = result.get("answer", "Could not generate summary.")
        st.write(answer)
        # Show guardrail alerts if any
        all_flags = result.get("input_flags", []) + result.get("output_flags", [])
        if all_flags:
            with st.expander("🛡️ Guardrail alerts", expanded=True):
                for flag in all_flags:
                    st.warning(flag)
        with st.expander("🔍 Graph execution trace"):
            st.code(result.get("node_trace", ""))
        if result.get("ranking_scores"):
            with st.expander("📊 Context ranking details"):                st.code(result.get("ranking_scores", ""))

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
            # Show guardrail flags if any
            all_flags = msg.get("input_flags", []) + msg.get("output_flags", [])
            if all_flags:
                with st.expander("🛡️ Guardrail alerts", expanded=True):
                    for flag in all_flags:
                        st.warning(flag)
            if msg.get("trace"):
                with st.expander("🔍 Graph trace for this answer", expanded=False):
                    st.code(msg["trace"])
            if msg.get("ranking_scores"):
                with st.expander("📊 Context ranking details", expanded=False):
                    st.code(msg["ranking_scores"])

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
            "input_flags": result.get("input_flags", []),
            "output_flags": result.get("output_flags", []),
            "ranking_scores": result.get("ranking_scores", ""),
        })

        st.session_state["chat_input_key"] += 1
        st.rerun()
