from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
from pinecone import Pinecone
from groq import Groq
from pypdf import PdfReader
import hashlib

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise RuntimeError("Missing required environment variables")

# Initialize Groq and Pinecone
groq_client = Groq(api_key=GROQ_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

# Helper function to get embeddings from Groq using text similarity
def get_simple_embedding(text: str):
    """Generate a simple embedding by hashing text"""
    # This is a placeholder - in production use a proper embedding model
    # For now, we'll use a simple hash-based approach
    hash_obj = hashlib.md5(text.encode())
    hash_hex = hash_obj.hexdigest()
    # Convert hex to float vector
    embedding = [float(int(hash_hex[i:i+2], 16)) / 255.0 for i in range(0, len(hash_hex), 2)]
    # Pad to 384 dimensions (Pinecone index dimension)
    while len(embedding) < 384:
        embedding.extend(embedding[:384-len(embedding)])
    return embedding[:384]

# Ingest endpoint
@app.post("/ingest")
def ingest_pdf():
    try:
        pdf_path = "data/domestic_care_services.pdf"
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF not found")
        
        logger.info(f"Loading PDF from {pdf_path}...")
        reader = PdfReader(pdf_path)
        
        documents = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append({
                    "text": text,
                    "page": page_num + 1,
                    "source": "domestic_care_services.pdf"
                })
        
        logger.info(f"Extracted {len(documents)} pages from PDF")
        
        # Chunk documents and create embeddings
        vectors_to_upsert = []
        for i, doc in enumerate(documents):
            embedding = get_simple_embedding(doc["text"])
            vector_id = f"doc_{i}"
            vectors_to_upsert.append((
                vector_id,
                embedding,
                {
                    "text": doc["text"][:500],  # Store first 500 chars
                    "page": doc["page"],
                    "source": doc["source"]
                }
            ))
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors_to_upsert)
        logger.info(f"Ingested {len(vectors_to_upsert)} vectors to Pinecone")
        
        return {
            "status": "success",
            "documents_ingested": len(documents),
            "vectors_created": len(vectors_to_upsert)
        }
    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/query")
def query_chatbot(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Empty question")
        
        # Get embedding for question
        question_embedding = get_simple_embedding(request.question)
        
        # Query Pinecone for relevant documents
        query_result = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract context from Pinecone results
        context = "\n".join([
            match["metadata"].get("text", "")
            for match in query_result.get("matches", [])
            if match.get("score", 0) > 0.1
        ])
        
        if not context:
            context = "No relevant documents found in the knowledge base."
        
        # Generate answer using Groq
        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {request.question}\n\nProvide a helpful answer based on the context."
                }
            ],
            max_tokens=1024,
        )
        
        answer = message.choices[0].message.content
        
        # Extract sources
        sources = list(set([
            match["metadata"].get("source", "Unknown")
            for match in query_result.get("matches", [])
        ]))
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
