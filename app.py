import streamlit as st
import os
import faiss
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF se Sawal-Jawab",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- CSS for Custom Styling (Updated) ---
st.markdown("""
<style>
    h1 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching Functions ---
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sbert_model()

@st.cache_resource
def create_vector_index(_pdf_bytes):
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(_pdf_bytes)
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    chunks = [doc.page_content for doc in docs]
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    os.remove(temp_file_path)
    return index, chunks

# --- Main App Logic ---

st.title("Muskabhishek ke PDF Dost se Guftagoo 💬")
st.write("---")

# API Key Configuration using Streamlit Secrets
try:
    api_key = st.secrets["GEMINI_API"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("Gemini API Key configure nahi ho paayi. Kripya Streamlit Cloud ke settings mein 'GEMINI_API_KEY' secret add karein.")
    st.stop()

# PDF File Uploader
uploaded_file = st.file_uploader("aug.pdf", type="pdf")
st.write("---")

if uploaded_file:
    with st.spinner("PDF ko process kiya ja raha hai... कृपया प्रतीक्षा करें..."):
        pdf_bytes = uploaded_file.getvalue()
        index, chunks = create_vector_index(pdf_bytes)
        st.success("PDF taiyaar hai! Ab aap sawal pooch sakte hain. 🎉")

    st.header("Apne PDF se sawal poochein")
    question = st.text_input("Sawal yahan likhein:", key="question_input")

    if st.button("Jawab Pata Karein"):
        if question:
            with st.spinner("Jawab dhoonda ja raha hai..."):
                try:
                    question_embedding = model.encode([question])
                    question_embedding_np = np.array(question_embedding, dtype=np.float32)
                    k = 5
                    distances, indices = index.search(question_embedding_np, k)
                    context = " ".join([chunks[i] for i in indices[0]])
                    
                    # YAHAN BADLAAV KIYA GAYA HAI
                    gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
                    
                    prompt = f"""Aap ek expert document analyst hain. Diye gaye context ke आधार par sawal ka sateek aur saaranshit jawab dein.
                    Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
                    response = gemini_model.generate_content(prompt)
                    st.subheader("Jawab:")
                    st.markdown(f"> {response.text}")
                except Exception as e:
                    st.error(f"Jawab generate karne mein ek error aayi: {e}")
        else:
            st.warning("Kripya pehle ek sawal likhein.")



