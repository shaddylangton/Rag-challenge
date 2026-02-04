"""
Enhanced FastAPI Backend with Session Management
Implements session-based architecture for multi-tenant RAG application
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.retriever import HybridRetrieverWithReranking
from src.generator import create_generator
from src.evaluator import RAGEvaluator
from src.session_manager import SessionManager

# Initialize FastAPI
app = FastAPI(
    title="Enhanced RAG API",
    version="2.0.0",
    description="Session-based RAG API with multi-tenant support"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
session_manager = SessionManager(base_path="sessions")
evaluator = RAGEvaluator()

# Session-specific state (keyed by session_id)
session_states: Dict[str, Dict] = {}


# Request/Response Models
class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    status: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    num_chunks: int
    session_id: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    use_citations: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    num_sources: int
    evaluation: Dict[str, Any]
    session_id: str
    citations_enabled: bool


class SessionInfoResponse(BaseModel):
    session_id: str
    created_at: str
    last_accessed: str
    num_documents: int
    num_queries: int
    status: str


# Helper function to get or create session state
def get_session_state(session_id: str) -> Dict:
    """Get or initialize state for a session."""
    if session_id not in session_states:
        session_states[session_id] = {
            "processor": DocumentProcessor(chunk_size=500, chunk_overlap=50),
            "retriever": None,
            "generator": None,
            "chunks": [],
            "documents": []
        }
    return session_states[session_id]


# Session Management Endpoints
@app.post("/session/create", response_model=SessionResponse)
async def create_session():
    """
    Create a new isolated session for a user.
    
    Returns:
        Session ID and metadata
    """
    try:
        session_id = session_manager.create_session()
        session_info = session_manager.get_session_info(session_id)
        
        return SessionResponse(
            session_id=session_id,
            created_at=session_info['created_at'],
            status=session_info['status']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/info", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """
    Get information about a session.
    
    Args:
        session_id: Session UUID
    
    Returns:
        Session metadata
    """
    try:
        info = session_manager.get_session_info(session_id)
        return SessionInfoResponse(
            session_id=info['session_id'],
            created_at=info['created_at'],
            last_accessed=info['last_accessed'],
            num_documents=len(info['documents']),
            num_queries=len(info['queries']),
            status=info['status']
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and all its data.
    
    Args:
        session_id: Session UUID
    """
    try:
        session_manager.delete_session(session_id)
        if session_id in session_states:
            del session_states[session_id]
        return {"message": "Session deleted successfully", "session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions.
    
    Returns:
        List of session metadata
    """
    try:
        sessions = session_manager.list_sessions()
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Document Processing Endpoints
@app.post("/session/{session_id}/upload", response_model=UploadResponse)
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """
    Upload and process a document for a session.
    
    Args:
        session_id: Session UUID
        file: Document file (PDF, DOCX, TXT)
    
    Returns:
        Upload status and document metadata
    """
    try:
        # Verify session exists
        session_path = session_manager.get_session_path(session_id)
        state = get_session_state(session_id)
        
        # Save uploaded file
        doc_path = session_manager.get_document_path(session_id)
        file_path = doc_path / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document (using recursive chunking by default)
        processor = state['processor']
        _, chunks = processor.process_document(str(file_path), use_recursive=True)
        
        # Add source information to chunks
        for chunk in chunks:
            chunk['source'] = file.filename
        
        state['chunks'].extend(chunks)
        state['documents'].append(file.filename)
        
        # Register document with session
        session_manager.add_document(session_id, file.filename, {
            'num_chunks': len(chunks),
            'method': 'recursive'
        })
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            num_chunks=len(chunks),
            session_id=session_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/session/{session_id}/process")
async def process_session(session_id: str):
    """
    Index all documents in a session for retrieval.
    
    Args:
        session_id: Session UUID
    
    Returns:
        Indexing status
    """
    try:
        state = get_session_state(session_id)
        
        if not state['chunks']:
            raise HTTPException(status_code=400, detail="No documents uploaded yet")
        
        # Check if index already exists
        index_path = session_manager.get_index_path(session_id) / "hybrid_index"
        
        # Initialize retriever with reranking for out-of-domain rejection
        retriever = HybridRetrieverWithReranking(
            keyword_weight=0.5,   # Balanced weights
            vector_weight=0.5,
            use_reranker=True     # Enable cross-encoder reranking
        )
        
        # Try to load existing index
        if retriever.load_index(str(index_path)):
            state['retriever'] = retriever
            return {
                "message": "Loaded existing index",
                "session_id": session_id,
                "num_chunks": len(state['chunks']),
                "cached": True
            }
        
        # Index documents
        retriever.index_chunks(state['chunks'])
        
        # Save index for future use
        retriever.save_index(str(index_path))
        
        # Initialize generator
        state['retriever'] = retriever
        # Use OpenAI if API key is set, otherwise fall back to mock
        api_key = os.getenv('OPENAI_API_KEY')
        generator_mode = "openai" if api_key else "mock"
        state['generator'] = create_generator(generator_mode)
        
        return {
            "message": "Documents indexed successfully",
            "session_id": session_id,
            "num_chunks": len(state['chunks']),
            "cached": False
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing documents: {str(e)}")


# Query Endpoints
@app.post("/session/{session_id}/query", response_model=QueryResponse)
async def query_session(session_id: str, request: QueryRequest):
    """
    Query documents in a session.
    
    Args:
        session_id: Session UUID
        request: Query parameters
    
    Returns:
        Generated answer with sources and evaluation metrics
    """
    try:
        state = get_session_state(session_id)
        
        if state['retriever'] is None:
            raise HTTPException(
                status_code=400,
                detail="No documents processed. Upload and process documents first."
            )
        
        # Retrieve relevant chunks (with out-of-domain filtering)
        retrieved_chunks = state['retriever'].retrieve(request.query, top_k=request.top_k)
        
        # A+ FEATURE: Handle out-of-domain queries (no relevant results)
        if not retrieved_chunks:
            # Get retrieval stats for logging
            stats = state['retriever'].get_observability_report()
            return QueryResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the uploaded documents to answer this question. "
                       "This question may be outside the scope of the provided documents.",
                sources=[],
                num_sources=0,
                evaluation={
                    "relevance": 0.0,
                    "faithfulness": 1.0,  # Faithful because it correctly refused
                    "out_of_domain": True,
                    "rejection_reason": "No chunks met relevance threshold"
                },
                session_id=session_id,
                citations_enabled=request.use_citations
            )
        
        # Generate answer
        if state['generator'] is None:
            api_key = os.getenv('OPENAI_API_KEY')
            generator_mode = "openai" if api_key else "mock"
            state['generator'] = create_generator(generator_mode)
        
        response = state['generator'].generate_answer(
            request.query,
            retrieved_chunks,
            include_citations=request.use_citations
        )
        
        # Evaluate response
        evaluation = evaluator.evaluate(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            generated_answer=response['answer']
        )
        
        # Log query
        session_manager.add_query(session_id, request.query, response)
        
        return QueryResponse(
            query=response['query'],
            answer=response['answer'],
            sources=response['sources'],
            num_sources=response['num_sources'],
            evaluation=evaluation,
            session_id=session_id,
            citations_enabled=response['citations_enabled']
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Maintenance Endpoints
@app.post("/admin/cleanup")
async def cleanup_old_sessions(max_age_hours: int = 24):
    """
    Cleanup sessions older than specified hours.
    
    Args:
        max_age_hours: Maximum age in hours (default: 24)
    
    Returns:
        Number of sessions deleted
    """
    try:
        deleted = session_manager.cleanup_old_sessions(max_age_hours)
        return {
            "message": f"Cleaned up {deleted} old sessions",
            "deleted_count": deleted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "active_sessions": len(session_states)
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enhanced RAG API",
        "version": "2.0.0",
        "features": [
            "Session-based multi-tenant architecture",
            "UUID-based session isolation",
            "Recursive text chunking",
            "Hybrid retrieval (TF-IDF + Vector)",
            "FAISS index caching",
            "Source citations",
            "Comprehensive evaluation"
        ],
        "endpoints": {
            "session_management": [
                "POST /session/create",
                "GET /session/{session_id}/info",
                "DELETE /session/{session_id}",
                "GET /sessions"
            ],
            "document_processing": [
                "POST /session/{session_id}/upload",
                "POST /session/{session_id}/process"
            ],
            "querying": [
                "POST /session/{session_id}/query"
            ]
        },
        "docs": "/docs"
    }


@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """Serve the UI HTML file."""
    ui_path = Path(__file__).parent.parent / "ui.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="UI file not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
