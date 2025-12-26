import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv

from azure.ai.openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# --------------------------------------------------
# Azure Clients
# --------------------------------------------------
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(title="Azure RAG Bot")

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def embed_text(text: str):
    response = openai_client.embeddings.create(
        model=AZURE_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def load_and_chunk_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() for page in reader.pages])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_text(text)


def index_chunks(chunks):
    documents = []

    for chunk in chunks:
        documents.append({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "vector": embed_text(chunk)
        })

    search_client.upload_documents(documents)


def retrieve_context(query: str, k: int = 3):
    query_vector = embed_text(query)

    results = search_client.search(
        search_text=None,
        vector_queries=[{
            "vector": query_vector,
            "k": k,
            "fields": "vector"
        }]
    )

    return " ".join([doc["content"] for doc in results])


def generate_answer(query: str):
    context = retrieve_context(query)

    response = openai_client.chat.completions.create(
        model=AZURE_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer ONLY using the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# --------------------------------------------------
# API Endpoints
# --------------------------------------------------
@app.post("/upload")
async def upload_document(file: UploadFile):
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_and_chunk_pdf(file_path)
    index_chunks(chunks)

    return {
        "status": "success",
        "chunks_indexed": len(chunks)
    }


@app.post("/chat")
async def chat(query: str):
    answer = generate_answer(query)
    return {"answer": answer}


# Run with:
# uvicorn main:app --reload
