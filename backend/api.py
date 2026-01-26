"""
FastAPI Backend for RAG Application
Exposes REST API endpoints for document processing, querying, and evaluation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_processor import DocumentProcessor
from retriever import HybridRetriever
from generator import create_generator
from evaluator import RAGEvaluator

# Initialize FastAPI
app = FastAPI(title="RAG API", version="1.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use database or cache)
rag_state = {
    "processor": DocumentProcessor(chunk_size=500, chunk_overlap=50),
    "retriever": None,
    "generator": None,
    "evaluator": RAGEvaluator(),
    "chunks": [],
    "document_name": None
}


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    evaluation: Dict[str, Any]


class DocumentInfo(BaseModel):
    name: str
    num_chunks: int
    total_chars: int


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "RAG API is running",
        "document_loaded": rag_state["document_name"] is not None
    }


@app.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    Supports: PDF, TXT, DOCX
    """
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Process document
        metadata, chunks = rag_state["processor"].process_document(tmp_path)
        
        # Index with retriever
        rag_state["retriever"] = HybridRetriever()
        rag_state["retriever"].index_chunks(chunks)
        
        # Initialize generator (using mock for demo, can use OpenAI)
        rag_state["generator"] = create_generator("mock")
        
        # Store state
        rag_state["chunks"] = chunks
        rag_state["document_name"] = file.filename
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        return DocumentInfo(
            name=file.filename,
            num_chunks=len(chunks),
            total_chars=metadata.get("length", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        file.file.close()


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the loaded document
    Returns: answer, retrieved chunks, evaluation metrics
    """
    if not rag_state["retriever"]:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a document first.")
    
    try:
        # Retrieve relevant chunks
        retrieved_chunks = rag_state["retriever"].retrieve(request.query, top_k=request.top_k)
        
        # Generate answer
        answer = rag_state["generator"].generate(request.query, retrieved_chunks)
        
        # Evaluate
        evaluation = rag_state["evaluator"].evaluate(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            generated_answer=answer
        )
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            evaluation=evaluation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/document/info")
async def get_document_info():
    """Get information about the currently loaded document"""
    if not rag_state["document_name"]:
        raise HTTPException(status_code=400, detail="No document loaded")
    
    return {
        "name": rag_state["document_name"],
        "num_chunks": len(rag_state["chunks"]),
        "sample_chunk": rag_state["chunks"][0]["text"][:200] if rag_state["chunks"] else None
    }


@app.delete("/document")
async def clear_document():
    """Clear the currently loaded document"""
    rag_state["retriever"] = None
    rag_state["generator"] = None
    rag_state["chunks"] = []
    rag_state["document_name"] = None
    
    return {"message": "Document cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
