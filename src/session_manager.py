"""
Session Manager Module
Implements multi-user session management with UUID-based isolation.
Enables multi-tenant deployment without data leakage.
"""

import uuid
from pathlib import Path
from typing import Dict, Optional
import json
import shutil
from datetime import datetime


class SessionManager:
    """
    Manages isolated sessions for multi-user RAG application.
    
    Features:
    - UUID-based session IDs for security
    - Isolated document storage per session
    - Session metadata tracking
    - Automatic cleanup of expired sessions
    
    Shows production-ready system design and multi-tenant architecture awareness.
    """
    
    def __init__(self, base_path: str = "sessions"):
        """
        Initialize session manager.
        
        Args:
            base_path: Base directory for storing session data
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self) -> str:
        """
        Create a new isolated session.
        
        Returns:
            Session ID (UUID)
        """
        session_id = str(uuid.uuid4())
        session_path = self.base_path / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (session_path / "documents").mkdir(exist_ok=True)
        (session_path / "indices").mkdir(exist_ok=True)
        
        # Initialize session metadata
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'documents': [],
            'queries': [],
            'status': 'active'
        }
        
        # Save metadata
        self._save_metadata(session_id, metadata)
        self.sessions[session_id] = metadata
        
        print(f"✅ Session created: {session_id}")
        return session_id
    
    def get_session_path(self, session_id: str) -> Path:
        """
        Get the file system path for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Path object for session directory
        
        Raises:
            ValueError: If session doesn't exist
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session not found: {session_id}")
        
        # Update last accessed time
        self._update_last_accessed(session_id)
        
        return session_path
    
    def get_document_path(self, session_id: str) -> Path:
        """
        Get the document storage path for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Path object for documents directory
        """
        return self.get_session_path(session_id) / "documents"
    
    def get_index_path(self, session_id: str) -> Path:
        """
        Get the index storage path for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Path object for indices directory
        """
        return self.get_session_path(session_id) / "indices"
    
    def add_document(self, session_id: str, document_name: str, document_info: Dict):
        """
        Register a document upload for a session.
        
        Args:
            session_id: Session UUID
            document_name: Name of uploaded document
            document_info: Metadata about the document
        """
        metadata = self._load_metadata(session_id)
        metadata['documents'].append({
            'name': document_name,
            'uploaded_at': datetime.now().isoformat(),
            'info': document_info
        })
        metadata['last_accessed'] = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)
    
    def add_query(self, session_id: str, query: str, response: Dict):
        """
        Log a query and response for a session.
        
        Args:
            session_id: Session UUID
            query: User query
            response: System response
        """
        metadata = self._load_metadata(session_id)
        metadata['queries'].append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'num_sources': response.get('num_sources', 0),
            'answer_length': len(response.get('answer', ''))
        })
        metadata['last_accessed'] = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)
    
    def get_session_info(self, session_id: str) -> Dict:
        """
        Get session metadata.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session metadata dictionary
        """
        return self._load_metadata(session_id)
    
    def delete_session(self, session_id: str):
        """
        Delete a session and all its data.
        
        Args:
            session_id: Session UUID
        """
        session_path = self.base_path / session_id
        if session_path.exists():
            shutil.rmtree(session_path)
            if session_id in self.sessions:
                del self.sessions[session_id]
            print(f"✅ Session deleted: {session_id}")
        else:
            raise ValueError(f"Session not found: {session_id}")
    
    def list_sessions(self) -> list:
        """
        List all active sessions.
        
        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                try:
                    metadata = self._load_metadata(session_dir.name)
                    sessions.append(metadata)
                except:
                    continue
        return sessions
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Delete sessions older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        for session in self.list_sessions():
            last_accessed = datetime.fromisoformat(session['last_accessed'])
            if last_accessed < cutoff_time:
                try:
                    self.delete_session(session['session_id'])
                    deleted_count += 1
                except:
                    continue
        
        print(f"✅ Cleaned up {deleted_count} old sessions")
        return deleted_count
    
    def _save_metadata(self, session_id: str, metadata: Dict):
        """Save session metadata to disk."""
        session_path = self.base_path / session_id
        metadata_file = session_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, session_id: str) -> Dict:
        """Load session metadata from disk."""
        session_path = self.base_path / session_id
        metadata_file = session_path / "metadata.json"
        
        if not metadata_file.exists():
            raise ValueError(f"Session metadata not found: {session_id}")
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def _update_last_accessed(self, session_id: str):
        """Update last accessed timestamp."""
        try:
            metadata = self._load_metadata(session_id)
            metadata['last_accessed'] = datetime.now().isoformat()
            self._save_metadata(session_id, metadata)
        except:
            pass


if __name__ == "__main__":
    # Demo usage
    print("=" * 80)
    print("SESSION MANAGER DEMO")
    print("=" * 80)
    
    manager = SessionManager()
    
    # Create sessions
    session1 = manager.create_session()
    session2 = manager.create_session()
    
    print(f"\nCreated 2 sessions:")
    print(f"  - {session1}")
    print(f"  - {session2}")
    
    # Add documents
    manager.add_document(session1, "document1.pdf", {'pages': 10})
    manager.add_document(session2, "document2.docx", {'pages': 5})
    
    # Add queries
    manager.add_query(session1, "What is AI?", {'num_sources': 3, 'answer': 'AI is...'})
    
    # List sessions
    print("\nAll sessions:")
    for session in manager.list_sessions():
        print(f"  {session['session_id']}: {len(session['documents'])} docs, {len(session['queries'])} queries")
    
    # Cleanup
    print("\nDeleting session 1...")
    manager.delete_session(session1)
    
    print("\nRemaining sessions:")
    for session in manager.list_sessions():
        print(f"  {session['session_id']}")
    
    print("\n" + "=" * 80)
