# Zaruri libraries import karein
import os
import faiss
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google.colab import userdata

# Apni Gemini API key ko Colab Secrets mein daalein
# Key ka naam 'GEMINI_API_KEY' hona chahiye
genai.configure(api_key=userdata.get("GEMINI_API"))

# Apni PDF file ko Colab mein upload karna mat bhulna
file_path = "aug.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

# Documents ko chunks mein vibhajit karein
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Chunks ko text mein badle
chunks = [doc.page_content for doc in docs]

# Embeddings banayein (S-BERT model se)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# FAISS index banayein
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Sawal-Jawab ka function
def ask_gemini_with_rag(question):
    # Sawal ki embedding banayein
    question_embedding = model.encode([question])

    # FAISS index mein sabse milte-julte chunks dhoondhein
    k = 3
    distances, indices = index.search(question_embedding, k)

    # Context ke liye milte-julte chunks ko jod dein
    context = ""
    for i in indices[0]:
        context += chunks[i] + " "

    # Gemini model ko load karein
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')

    # Prompt banayein
    prompt = f"Given the following context, answer the question accurately and concisely.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    # Gemini se jawab maangein
    response = gemini_model.generate_content(prompt)

    return response.text

# Sawal poochein
question = "hi"
answer = ask_gemini_with_rag(question)

print(f"Question: {question}")
print("---")
print(f"Answer: {answer}")