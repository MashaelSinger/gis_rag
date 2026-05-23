import streamlit as st
import chromadb
# Force clear the system cache to prevent stale tenant connections
chromadb.api.client.SharedSystemClient.clear_system_cache()
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types

# --- UI TRANSLATION DATA ---
UI = {
    "English": {
        "title": "GIS Document Assistant (Fast Mode)",
        "doc_hdr": "Data Ingestion (Local + Uploads)",
        "btn_process": "Process Documents",
        "input_ph": "Ask a question about your GIS documents...",
        "status_proc": "Processing locally... (Instant indexing)",
        "err_key": "Please provide your Google API Key (for chat only).",
        "success": "Knowledge base indexed successfully!"
    },
    "العربية": {
        "title": "مساعد مستندات GIS (وضع السرعة)",
        "doc_hdr": "إضافة المستندات (مجلد محلّي + رفع)",
        "btn_process": "معالجة المستندات",
        "input_ph": "اطرح سؤالاً حول مستنداتك...",
        "status_proc": "جاري الفهرسة محلياً... (سريع جداً)",
        "err_key": "يرجى إدخال مفتاح الـ API.",
        "success": "تمت فهرسة المستندات بنجاح!"
    }
}

st.set_page_config(page_title="GIS Assistant", layout="wide")

# --- INITIALIZATION ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vstore" not in st.session_state: st.session_state.vstore = None

with st.sidebar:
    lang = st.radio("Language / اللغة", ["English", "العربية"])
    txt = UI[lang]
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    st.header("📂 Current File Inventory")
    if os.path.exists("data"):
        files = [f for f in os.listdir("data") if f.endswith(".pdf")]
        for f in files: st.caption(f"📁 {f}")

st.title(txt["title"])

# --- FAST LOCAL EMBEDDING ENGINE ---
# This runs on YOUR computer, not Google's server
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- DATA INGESTION ENGINE ---
uploaded_files = st.file_uploader(txt["doc_hdr"], type="pdf", accept_multiple_files=True)

if st.button(txt["btn_process"]):
    if not api_key: st.error(txt["err_key"])
    else:
        with st.spinner(txt["status_proc"]):
            chunks = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            
            # Scan Local 'data' folder
            if os.path.exists("data"):
                for filename in os.listdir("data"):
                    if filename.endswith(".pdf"):
                        reader = PdfReader(os.path.join("data", filename))
                        for i, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                for c in splitter.split_text(text):
                                    chunks.append(Document(page_content=c, metadata={"source": filename, "page": i+1}))
            
            # Scan UI Uploads
            if uploaded_files:
                for f in uploaded_files:
                    reader = PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            for c in splitter.split_text(text):
                                chunks.append(Document(page_content=c, metadata={"source": f.name, "page": i+1}))
            
            if not chunks: st.warning("No PDFs found.")
            else:
                # Use the fast local embedder
                st.session_state.vstore = Chroma.from_documents(chunks, get_embeddings())
                st.success(txt["success"])

# --- CHAT INTERFACE ---
if st.session_state.vstore:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["text"])
    
    if prompt := st.chat_input(txt["input_ph"]):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "text": prompt})
        
        with st.chat_message("assistant"):
            docs = st.session_state.vstore.similarity_search(prompt, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # Still use Google Gemini for the intelligent chat response
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=f"Context: {context}\n\nQuestion: {prompt}",
            )
            st.markdown(response.text)
            st.session_state.chat_history.append({"role": "assistant", "text": response.text})