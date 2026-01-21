import os
import shutil
import uuid
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_CHAT_MODEL = os.getenv("AZURE_CHAT_MODEL", "gpt-4o")
AZURE_EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL", "text-embedding-3-small")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "rag-index")

# Initialize Clients
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# --- SESSION STORAGE ---
chat_sessions = {}

# --- HELPER FUNCTIONS ---

def embed_text(text: str):
    response = openai_client.embeddings.create(
        model=AZURE_EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def load_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() for page in reader.pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def load_word(file_path: str):
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_text(text)
    except Exception as e:
        print(f"Error reading Word doc: {e}")
        return []

def load_excel(file_path: str):
    try:
        df = pd.read_excel(file_path)
        # Convert to markdown for LLM readability
        text_content = df.to_markdown(index=False)
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        return splitter.split_text(text_content)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return []

def index_chunks(chunks, doc_id: str, source_filename: str):
    documents = []
    for chunk in chunks:
        documents.append({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "vector": embed_text(chunk),
            "doc_id": doc_id,
            "source": source_filename
        })
    search_client.upload_documents(documents)

def retrieve_context(query: str, k: int = 5):
    query_vector = embed_text(query)
    results = search_client.search(
        search_text=query,
        vector_queries=[{
            "kind": "vector",
            "vector": query_vector,
            "k": k,
            "fields": "vector"
        }]
    )
    return " ".join([f"[Source: {doc.get('source', 'unknown')}]: {doc['content']}" for doc in results])

# --- CORE LOGIC FUNCTIONS ---

async def handle_document_upload(file: UploadFile, user_id: Optional[str], is_transcript: bool) -> Dict[str, Any]:
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{uuid.uuid4()}_{file.filename}"
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize session if new
    if not user_id:
        user_id = str(uuid.uuid4())
        chat_sessions[user_id] = {
            "history": [],
            "current_precis": "",
            "files": []
        }

    doc_id = str(uuid.uuid4())
    chunks = []

    # Universal File Handler
    filename_lower = file.filename.lower()
    
    if filename_lower.endswith(".pdf"):
        chunks = load_pdf(file_path)
    elif filename_lower.endswith((".docx", ".doc")):
        chunks = load_word(file_path)
    elif filename_lower.endswith((".xlsx", ".xls")):
        chunks = load_excel(file_path)
    
    file_type = "transcript" if is_transcript else "extension"
    
    if chunks:
        index_chunks(chunks, doc_id, file.filename)
        
        # Track file in session
        if user_id in chat_sessions:
            chat_sessions[user_id]["files"].append({
                "filename": file.filename,
                "doc_id": doc_id,
                "type": file_type
            })

    return {
        "status": "success",
        "user_id": user_id,
        "file_type": file_type, 
        "message": f"Processed {file.filename}. You can now ask to {'generate' if is_transcript else 'update'} the precis."
    }

def handle_chat_generation(user_id: str, query: str) -> Dict[str, Any]:
    if user_id not in chat_sessions:
        return {"error": "Session not found."}

    session = chat_sessions[user_id]
    
    # 1. Retrieve Context
    context = retrieve_context(query)
    
    # 2. Construct Prompt
    current_precis = session.get("current_precis", "")
    
    system_prompt = """
    You are an expert editor. You maintain a living document called a 'Precis'.
    
    YOUR GOAL:
    - If the user asks to create the summary/precis, write it based on the context.
    - If the user asks to UPDATE or EXTEND the precis with new information, REWRITE the 'Current Draft Precis' to include the new insights found in the context.
    
    INPUT DATA:
    - You have access to PDFs, Word Docs, and Excel sheets uploaded by the user.
    - Treat all of them as factual sources.
    
    OUTPUT FORMAT:
    Return your response in two clearly separated parts:
    [ANSWER]: A short conversational confirmation (e.g., "I've added the new market research data to the summary.")
    [PRECIS]: The full, updated text of the precis. (Always return the full text if you modify it).
    """

    user_message = f"""
    --- CURRENT DRAFT PRECIS ---
    {current_precis}
    
    --- RETRIEVED CONTEXT FROM FILES ---
    {context}
    
    --- USER INSTRUCTION ---
    {query}
    """

    session["history"].append({"role": "user", "content": query})

    response = openai_client.chat.completions.create(
        model=AZURE_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2
    )

    answer_full = response.choices[0].message.content
    
    # Logic to separate Answer from Precis
    conversational_reply = answer_full
    if "[PRECIS]" in answer_full:
        parts = answer_full.split("[PRECIS]")
        conversational_reply = parts[0].replace("[ANSWER]:", "").strip()
        new_precis = parts[1].strip()
        session["current_precis"] = new_precis
    
    session["history"].append({"role": "assistant", "content": conversational_reply})

    return {
        "answer": conversational_reply,
        "current_precis": session.get("current_precis"),
        "has_charts": "chart" in query.lower()
    }


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional

# Import the logic functions from upload.py
from upload import handle_document_upload, handle_chat_generation

app = FastAPI()

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(None), 
    is_transcript: bool = Form(True) 
):
    """
    Endpoint to handle file uploads (PDF, DOCX, XLSX).
    Delegates logic to upload.py
    """
    try:
        result = await handle_document_upload(file, user_id, is_transcript)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def generate_answer(user_id: str, query: str):
    """
    Endpoint to handle chat interactions and precis generation.
    Delegates logic to upload.py
    """
    try:
        result = handle_chat_generation(user_id, query)
        if "error" in result:
             raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))