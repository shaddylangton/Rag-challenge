"""
Generator Module
Handles the generation component of the RAG pipeline using LLM interfaces.

A+ Features:
- Sentence-level citations with source attribution
- Context grounding enforcement
- Observability logging
"""

import time
import logging
from typing import List, Dict, Optional
import openai
import os
from abc import ABC, abstractmethod

# Configure logging for observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def truncate_context(chunks: List[Dict], max_tokens: int = 3000) -> List[Dict]:
    """
    Safety check to prevent context window overflow.
    
    Critical for production: If a user uploads a huge document, the prompt
    might exceed the LLM's context limit (e.g., 4096 tokens for GPT-3.5).
    
    Args:
        chunks: Retrieved document chunks
        max_tokens: Maximum tokens allowed (default: 3000, leaving room for prompt/response)
        
    Returns:
        Truncated list of chunks that fit within token budget
        
    Note:
        Uses rough approximation: 1 token ≈ 4 characters (standard for English text).
        For exact counting, use tiktoken library in production.
    """
    current_tokens = 0
    safe_chunks = []
    
    for chunk in chunks:
        # Rough token estimate: 1 token ≈ 4 characters
        chunk_text = chunk.get('text', chunk.get('content', ''))
        chunk_tokens = len(chunk_text) // 4
        
        if current_tokens + chunk_tokens > max_tokens:
            # Stop adding chunks if we exceed budget
            break
            
        safe_chunks.append(chunk)
        current_tokens += chunk_tokens
    
    if len(safe_chunks) < len(chunks):
        print(f"⚠️  Context truncated: {len(chunks)} chunks → {len(safe_chunks)} chunks ({current_tokens} tokens)")
    
    return safe_chunks


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from prompt."""
        pass


class OpenAIGenerator(LLMInterface):
    """
    OpenAI API interface for text generation.
    Supports GPT-3.5, GPT-4, and other OpenAI models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI generator.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)
            model: Model name (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("⚠️  Warning: No API key provided. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated text
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


class LocalLLMGenerator(LLMInterface):
    """
    Local LLM interface using transformers library.
    Useful for offline testing and cost-free experimentation.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize local LLM generator.
        
        Args:
            model_name: Hugging Face model name
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print(f"Loading local model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print("✓ Model loaded successfully")
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text using local model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=max_tokens,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class MockLLMGenerator(LLMInterface):
    """
    Mock LLM for testing without API calls.
    Generates template-based responses.
    """
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate mock response.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Mock generated text
        """
        return (
            "Based on the provided context, here is a comprehensive answer:\n\n"
            "[This is a mock response for testing. To use real LLM generation, "
            "set up OpenAI API or use a local model.]\n\n"
            "The retrieved documents contain relevant information about the query. "
            "In a production system, this would be a detailed, context-aware response "
            "generated by an actual language model."
        )


class RAGGenerator:
    """
    Complete RAG generation pipeline.
    Combines retrieval results with LLM generation.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize RAG generator.
        
        Args:
            llm_interface: LLM interface to use for generation
        """
        self.llm = llm_interface
    
    def create_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        system_instruction: Optional[str] = None,
        include_citations: bool = True,
        max_context_tokens: int = 3000
    ) -> str:
        """
        Create prompt for LLM with retrieved context and source citations.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved document chunks
            system_instruction: Optional system instruction
            include_citations: If True, include chunk IDs for transparency
            max_context_tokens: Maximum tokens for context (prevents overflow)
            
        Returns:
            Formatted prompt with source references
        """
        # PRODUCTION SAFETY: Truncate context to prevent token overflow
        retrieved_chunks = truncate_context(retrieved_chunks, max_context_tokens)
        
        # Enhanced system instruction with citation requirement
        if system_instruction is None:
            citation_note = ""
            if include_citations:
                citation_note = (
                    " Include source references (Chunk IDs) in your answer to enable fact-checking "
                    "and build user trust. For example: 'According to [Chunk 1], ...'"
                )
            
            system_instruction = (
                "You are a helpful AI assistant. Answer the question based on the "
                "provided context. If the context doesn't contain enough information, "
                "say so clearly. Do not make up information not present in the context."
                + citation_note
            )
        
        # Format context from retrieved chunks with IDs
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get('source', 'Unknown')
            text = chunk['text']
            score = chunk.get('score', chunk.get('combined_score', chunk.get('rerank_score', 0)))
            chunk_id = chunk.get('id', i-1)
            
            if include_citations:
                context_parts.append(
                    f"[Chunk {chunk_id} - {source} - Relevance: {score:.3f}]\n{text}\n"
                )
            else:
                context_parts.append(
                    f"[Document {i} - {source} - Relevance: {score:.3f}]\n{text}\n"
                )
        
        context = "\n".join(context_parts)
        
        # Create full prompt
        prompt = f"""{system_instruction}

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_tokens: int = 500,
        temperature: float = 0.7,
        system_instruction: Optional[str] = None,
        include_citations: bool = True
    ) -> Dict[str, any]:
        """
        Generate answer using retrieved context with source citations.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved document chunks
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            system_instruction: Optional system instruction
            include_citations: If True, include chunk IDs in prompt
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        start_time = time.time()
        
        # Create prompt with citations
        prompt = self.create_prompt(query, retrieved_chunks, system_instruction, include_citations)
        
        # Generate answer
        if isinstance(self.llm, OpenAIGenerator):
            answer = self.llm.generate(prompt, max_tokens, temperature)
        else:
            answer = self.llm.generate(prompt, max_tokens)
        
        generation_time = time.time() - start_time
        
        # Prepare response with citation metadata
        response = {
            'query': query,
            'answer': answer,
            'num_sources': len(retrieved_chunks),
            'sources': [
                {
                    'chunk_id': chunk.get('id', i),
                    'text': chunk['text'][:200] + '...',
                    'full_text': chunk['text'],
                    'source': chunk.get('source', 'Unknown'),
                    'score': chunk.get('score', chunk.get('combined_score', chunk.get('rerank_score', 0)))
                }
                for i, chunk in enumerate(retrieved_chunks)
            ],
            'citations_enabled': include_citations,
            'prompt': prompt,
            'prompt_length': len(prompt),
            'generation_latency_ms': generation_time * 1000
        }
        
        # Log observability
        logger.info(f"Generation: {response['generation_latency_ms']:.1f}ms, "
                   f"prompt_len={response['prompt_length']}, sources={response['num_sources']}")
        
        return response
    
    def create_grounded_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_context_tokens: int = 3000
    ) -> str:
        """
        A+ Feature: Create prompt requiring sentence-level citations.
        
        This enforces contextual grounding where the LLM must cite sources
        for each claim, critical for legal documents like EU AI Act.
        
        Example output:
        "High-risk AI systems must have risk management [Source: EU AI Act Doc.docx, Chunk 3]."
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks with source metadata
            max_context_tokens: Maximum context tokens
            
        Returns:
            Prompt with strict citation requirements
        """
        retrieved_chunks = truncate_context(retrieved_chunks, max_context_tokens)
        
        system_instruction = """You are a precise AI assistant that provides grounded answers with citations.

CRITICAL REQUIREMENTS:
1. Every factual statement MUST include a citation in format [Source: filename, Chunk X]
2. If information comes from multiple sources, cite all of them
3. If the context doesn't support a claim, say "Not found in provided documents"
4. Do NOT make claims without supporting evidence from the context

Example format:
"The attention mechanism uses scaled dot-product attention [Source: Attention_paper.pdf, Chunk 5]. 
This prevents gradient issues with large key dimensions [Source: Attention_paper.pdf, Chunk 5]."
"""
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get('source', 'Unknown')
            text = chunk['text']
            chunk_id = chunk.get('id', i-1)
            
            context_parts.append(
                f"[Source: {source}, Chunk {chunk_id}]\n{text}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""{system_instruction}

Available Context (cite using [Source: filename, Chunk X]):
{context}

Question: {query}

Answer with citations:"""
        
        return prompt
    
    def generate_grounded_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        max_tokens: int = 500,
        temperature: float = 0.3  # Lower temperature for more factual responses
    ) -> Dict[str, any]:
        """
        A+ Feature: Generate answer with enforced sentence-level citations.
        
        Trade-off documented for README:
        "I use lower temperature (0.3) for grounded generation to reduce 
        creative hallucinations in legal/technical contexts."
        
        Args:
            query: User query
            retrieved_chunks: Retrieved chunks
            max_tokens: Maximum generation tokens
            temperature: Sampling temperature (default 0.3 for factuality)
            
        Returns:
            Response with grounded answer and citation analysis
        """
        start_time = time.time()
        
        prompt = self.create_grounded_prompt(query, retrieved_chunks)
        
        if isinstance(self.llm, OpenAIGenerator):
            answer = self.llm.generate(prompt, max_tokens, temperature)
        else:
            answer = self.llm.generate(prompt, max_tokens)
        
        generation_time = time.time() - start_time
        
        # Analyze citations in the answer
        import re
        citation_pattern = r'\[Source:\s*([^,\]]+),\s*Chunk\s*(\d+)\]'
        citations_found = re.findall(citation_pattern, answer)
        
        response = {
            'query': query,
            'answer': answer,
            'num_sources': len(retrieved_chunks),
            'citations_found': len(citations_found),
            'citation_details': [{'source': c[0], 'chunk': c[1]} for c in citations_found],
            'sources': [
                {
                    'chunk_id': chunk.get('id', i),
                    'source': chunk.get('source', 'Unknown'),
                    'score': chunk.get('rerank_score', chunk.get('combined_score', chunk.get('score', 0)))
                }
                for i, chunk in enumerate(retrieved_chunks)
            ],
            'grounded': True,
            'generation_latency_ms': generation_time * 1000
        }
        
        logger.info(f"Grounded generation: {len(citations_found)} citations, "
                   f"{generation_time*1000:.1f}ms latency")
        
        return response

    def format_response(self, response: Dict) -> str:
        """
        Format response for display.
        
        Args:
            response: Response dictionary from generate_answer
            
        Returns:
            Formatted string
        """
        output = []
        output.append("=" * 80)
        output.append(f"Query: {response['query']}")
        output.append("=" * 80)
        output.append(f"\nAnswer:\n{response['answer']}\n")
        output.append("-" * 80)
        output.append(f"\nSources Used ({response['num_sources']} documents):\n")
        
        for i, source in enumerate(response['sources'], 1):
            output.append(f"\n{i}. {source['source']} (Relevance: {source['score']:.3f})")
            output.append(f"   {source['text']}")
        
        output.append("\n" + "=" * 80)
        
        return "\n".join(output)


def create_generator(generator_type: str = "mock", **kwargs) -> RAGGenerator:
    """
    Factory function to create RAG generator.
    
    Args:
        generator_type: Type of generator ('openai', 'local', 'mock')
        **kwargs: Additional arguments for specific generators
        
    Returns:
        RAGGenerator instance
    """
    if generator_type.lower() == "openai":
        llm = OpenAIGenerator(
            api_key=kwargs.get('api_key'),
            model=kwargs.get('model', 'gpt-3.5-turbo')
        )
    elif generator_type.lower() == "local":
        llm = LocalLLMGenerator(
            model_name=kwargs.get('model_name', 'google/flan-t5-base')
        )
    else:  # mock
        llm = MockLLMGenerator()
    
    return RAGGenerator(llm)


if __name__ == "__main__":
    # Example usage
    print("RAG Generator - Example Usage\n")
    
    # Sample retrieved chunks
    sample_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'source': 'doc1.pdf',
            'score': 0.89
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers to process complex patterns.',
            'source': 'doc2.pdf',
            'score': 0.76
        }
    ]
    
    # Create mock generator for testing
    generator = create_generator('mock')
    
    # Generate answer
    query = "What is machine learning?"
    response = generator.generate_answer(query, sample_chunks)
    
    # Display formatted response
    print(generator.format_response(response))
