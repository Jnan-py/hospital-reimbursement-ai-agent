import streamlit as st
import pandas as pd
import json
import os
import PyPDF2
import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from streamlit_option_menu import option_menu


load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_KEY or not PINECONE_KEY:
    st.error("❗ Please set GOOGLE_API_KEY and PINECONE_API_KEY in a .env file or environment variables.")
    st.stop()

genai.configure(api_key=GOOGLE_KEY)


INDEX_NAME = "hospital-reimbursement-db"
pc = Pinecone(api_key=PINECONE_KEY)

# Create the index if it doesn't already exist
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def extract_text_chunks(uploaded_file) -> list[str]:

    filename = uploaded_file.name.lower()
    chunks: list[str] = []

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        for _, row in df.iterrows():
            parts = [f"{col}: {row[col]}" for col in df.columns]
            chunks.append(" | ".join(parts))

    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        for _, row in df.iterrows():
            parts = [f"{col}: {row[col]}" for col in df.columns]
            chunks.append(" | ".join(parts))

    elif filename.endswith(".json"):
        try:
            data = json.load(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            data = json.loads(uploaded_file.read().decode("utf-8"))

        if isinstance(data, list):
            for record in data:
                chunks.append(json.dumps(record, separators=(",", ": ")))
        elif isinstance(data, dict):
            flat = [f"{k}: {v}" for k, v in data.items()]
            chunks.append(" | ".join(flat))
        else:
            chunks.append(json.dumps(data, separators=(",", ": ")))

    elif filename.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks.extend(paras)

    elif filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        full_text = []
        for page in pdf_reader.pages:
            txt = page.extract_text() or ""
            full_text.append(txt)
        combined = "\n".join(full_text)
        paras = [p.strip() for p in combined.split("\n\n") if p.strip()]
        chunks.extend(paras)

    else:
        st.error(f"Unsupported file type: {filename}.")
        return []

    if not chunks:
        try:
            raw = uploaded_file.read().decode("utf-8", errors="ignore")
            if raw:
                chunks.append(raw)
        except Exception:
            pass

    return chunks


def build_vector_store_from_chunks(chunks: list[str]) -> PineconeVectorStore:

    index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    vector_store.add_texts(chunks)
    return vector_store


def get_relevant_context(query: str, vector_store: PineconeVectorStore, k: int = 3) -> list[str]:

    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def generate_bot_response(
    model: genai.GenerativeModel,
    query: str,
    context_chunks: list[str],
    history: list[dict],
) -> str:
    """
    Build a plain-text prompt that includes context and chat history, then return
    Gemini's reply as plain text (no JSON structure).
    """
    context_text = "\n".join(context_chunks)
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])

    prompt = f"""
You are an expert in medical billing and reimbursement. Use the context below to answer
the user's question clearly as plain text.

Context from the tariff:
{context_text}

Chat history:
{history_text}

New user query:
{query}

Answer in plain text, providing reimbursement rates, references, requirements, or exceptions
as needed, but do not output any JSON or special formatting—just plain English.
"""
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.2),
    )
    return response.text.strip()


def generate_quick_questions_via_gemini(chunks: list[str], num_samples: int = 10) -> list[str]:

    sample_chunks = chunks[:num_samples]
    sample_text = "\n".join(f"- {c}" for c in sample_chunks)
    prompt = f"""
You are an AI assistant that generates concise, relevant quick questions for hospital staff to ask
an automated reimbursement agent, based on these text snippets:

{sample_text}

Please produce a list of 5 distinct, user-friendly questions that someone might ask,
each on its own line, without numbering. Output plain text only.
"""
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction="Generate useful quick questions for reimbursement data."
    )
    resp = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.3),
    )
    raw_lines = resp.text.strip().split("\n")
    questions = []
    for line in raw_lines:
        txt = line.strip()
        txt = txt.lstrip("0123456789. ").strip()
        if txt:
            questions.append(txt)
    return questions[:5]


def delete_all_pinecone_vectors():

    index = pc.Index(INDEX_NAME)
    index.delete(delete_all=True)


st.set_page_config(page_title="Hospital Reimbursement AI Agent", layout="wide")
st.title("🏥 Hospital Reimbursement AI Agent")

if "raw_chunks" not in st.session_state:
    st.session_state.raw_chunks: list[str] = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "needs_index" not in st.session_state:
    st.session_state.needs_index = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "ask_history" not in st.session_state:
    st.session_state.ask_history: list[dict] = []
if "quick_questions" not in st.session_state:
    st.session_state.quick_questions: list[str] = []
if "selected_quick_ask" not in st.session_state:
    st.session_state.selected_quick_ask: str = ""

with st.sidebar:
    st.header("Upload Tariff/File")
    uploaded_file = st.file_uploader(
        "Any: .xlsx, .csv, .json, .txt, .pdf",
        type=["xlsx", "xls", "csv", "json", "txt", "pdf"],
    )

    if uploaded_file is not None and st.button("Load & Extract"):
        try:
            chunks = extract_text_chunks(uploaded_file)
            if not chunks:
                st.error("❗ Could not extract text from the file.")
            else:
                st.session_state.raw_chunks = chunks
                st.session_state.needs_index = True
                st.session_state.vector_store = None

                with st.spinner("Generating quick questions..."):
                    qs = generate_quick_questions_via_gemini(chunks, num_samples=10)
                st.session_state.quick_questions = qs

                st.session_state.chat_history = []
                st.session_state.ask_history = []
                st.success("✅ File loaded and extracted. Quick questions ready.")
        except Exception as e:
            st.error(f"❗ Error extracting text: {e}")

    st.markdown("---")

    st.header("Reset Session")
    if st.button("🔄 Reset Everything"):
        delete_all_pinecone_vectors()
        for key in [
            "raw_chunks",
            "vector_store",
            "needs_index",
            "chat_history",
            "ask_history",
            "quick_questions",
            "selected_quick_ask",
        ]:
            if isinstance(st.session_state.get(key), list):
                st.session_state[key] = []
            else:
                st.session_state[key] = None
        st.success("♻️ Session reset. Pinecone index cleared.")
        st.rerun()

mode = option_menu(None, ["Chat", "Ask Query"], orientation="horizontal", icons=["chat", "question-circle"], default_index=0)

if mode == "Ask Query":
    st.markdown("---")
    st.header("Recent Q&A")
    if st.session_state.ask_history:
        with st.expander("Show recent queries"):
            for qa in reversed(st.session_state.ask_history):
                st.markdown(f"**Q:** {qa['question']}")
                st.markdown(f"**A:** {qa['answer']}")
                st.markdown("---")
    else:
        st.write("No queries yet.")

if mode == "Chat":

    st.subheader("Chat")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    if st.session_state.raw_chunks:
        user_input = st.chat_input("Type your question here...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            if st.session_state.vector_store is None and st.session_state.needs_index:
                with st.spinner("Indexing data (once)—this may take 10–20s..."):
                    vs = build_vector_store_from_chunks(st.session_state.raw_chunks)
                    st.session_state.vector_store = vs
                    st.session_state.needs_index = False

            contexts = get_relevant_context(user_input, st.session_state.vector_store, k=3)
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction="You are an expert in medical billing and reimbursement."
            )
            with st.spinner("Generating response..."):
                answer = generate_bot_response(
                    model, user_input, contexts, st.session_state.chat_history
                )

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
    else:
        st.info("⬆️ Please upload a file (any supported type) to start chatting.")  


elif mode == "Ask Query":
    st.subheader("Ask a Single Query")

    if st.session_state.raw_chunks:
        default_input = st.session_state.selected_quick_ask
        question = st.text_input("Type your question here...", value=default_input)
        if st.button("Submit"):
            st.session_state.selected_quick_ask = ""
            st.session_state.ask_history.append({"question": question, "answer": ""})

            if st.session_state.vector_store is None and st.session_state.needs_index:
                with st.spinner("Indexing data (once)—this may take 10–20s..."):
                    vs = build_vector_store_from_chunks(st.session_state.raw_chunks)
                    st.session_state.vector_store = vs
                    st.session_state.needs_index = False

            contexts = get_relevant_context(question, st.session_state.vector_store, k=3)
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction="You are an expert in medical billing and reimbursement."
            )
            with st.spinner("Generating answer..."):
                ans = generate_bot_response(model, question, contexts, [{"role": "user", "content": question}])
            st.session_state.ask_history[-1]["answer"] = ans
            st.rerun()
    
        if st.session_state.ask_history:
            for idx, qa in enumerate(st.session_state.ask_history, start=1):
                st.markdown(f"**Q {idx}:** {qa['question']}")
                st.markdown(f"{qa['answer']}")
                st.markdown("---")
        else:
            st.write("No queries yet.")

        st.markdown("---")
        st.subheader("Quick Questions")
        if st.session_state.quick_questions:
            for i, q in enumerate(st.session_state.quick_questions):
                if st.button(f"{q}", key=f"quick_ask_{i}"):
                    st.session_state.selected_quick_ask = q
                    st.rerun()
        else:
            st.write("Upload a file to see quick questions.")

    else:
        st.info("Please upload a file (any supported type) to start asking queries.")
