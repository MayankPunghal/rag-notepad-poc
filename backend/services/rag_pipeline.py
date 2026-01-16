"""
RAG Pipeline module for RAG Brain
Orchestrates the complete retrieval-augmented generation pipeline
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

import numpy as np

import time

from config.settings import settings
from backend.core.database import (
    Document, DocumentRepository, Chunk, ChunkRepository, TagRepository,
    ConversationRepository, get_db_session
)
from backend.core.vector_store import get_vector_store, DualVectorStore
from backend.models.embeddings import get_text_embedding_model
from backend.services.text_processor import DocumentProcessor
from backend.services.mlflow_tracker import get_mlflow_tracker


# ============================================================================
# RAG Pipeline
# ============================================================================

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline
    Orchestrates ingestion, retrieval, and generation
    """

    def __init__(self):
        """Initialize RAG pipeline components"""
        self.vector_store: DualVectorStore = get_vector_store()
        self.embedding_model = get_text_embedding_model()
        self.document_processor = DocumentProcessor()
        self._chatbot = None  # Lazy-loaded to avoid circular import
        self.mlflow_tracker = get_mlflow_tracker()  # MLflow tracking

    @property
    def chatbot(self):
        """Get the chatbot instance (lazy-loaded to avoid circular import)"""
        if self._chatbot is None:
            from backend.services.glm_chatbot import get_chatbot
            self._chatbot = get_chatbot()
        return self._chatbot

    # ========================================================================
    # Ingestion Pipeline
    # ========================================================================

    async def ingest_text(
        self,
        text: str,
        filename: str,
        source: str = "upload",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a text document into the RAG system

        Args:
            text: Document text content
            filename: Original filename
            source: Source of the document
            metadata: Optional additional metadata

        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()

        # Generate document ID
        doc_id = Document.generate_doc_id()

        # Process the document
        result = self.document_processor.process_text_document(text, doc_id, filename)

        # Save to database
        async with get_db_session() as session:
            # Create document record
            doc = await DocumentRepository.create(
                session,
                doc_id=doc_id,
                filename=filename,
                content_type="text",
                file_path="",  # Will be set if file is saved
                raw_text=text,
                summary=result["summary"],
                source=source,
            )

            # Create chunk records
            for i, chunk in enumerate(result["chunks"]):
                chunk_record = await ChunkRepository.create(
                    session,
                    chunk_id=chunk["chunk_id"],
                    doc_id=doc_id,
                    content=chunk["content"],
                    faiss_index=self.vector_store.text_store.next_index + i,
                    chunk_number=chunk["chunk_number"],
                    token_count=chunk["token_count"],
                )

            # Add tags
            for tag_name, score in result["tags"]:
                await TagRepository.add_tag_to_document(
                    session, doc_id, tag_name, score, "auto"
                )

        # Add embeddings to FAISS
        chunk_ids = [c["chunk_id"] for c in result["chunks"]]
        self.vector_store.add_text_embeddings(result["embeddings"], chunk_ids)

        # Save vector store
        self.vector_store.save_all()

        processing_time = time.time() - start_time

        # Track with MLflow
        total_tokens = sum(c["token_count"] for c in result["chunks"])
        self.mlflow_tracker.track_ingestion(
            doc_type="text",
            doc_count=1,
            chunk_count=result["chunk_count"],
            total_tokens=total_tokens,
            processing_time=processing_time,
            filename=filename
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "ingested",
            "chunk_count": result["chunk_count"],
            "summary": result["summary"],
            "tags": [{"name": t[0], "score": t[1]} for t in result["tags"]],
            "created_at": datetime.utcnow().isoformat(),
        }

    async def ingest_image(
        self,
        image_bytes: bytes,
        filename: str,
        caption: str,
        detailed_info: Dict[str, str],
        source: str = "upload",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest an image into the RAG system with AI-powered description and tags

        Args:
            image_bytes: Image file bytes
            filename: Original filename
            caption: Generated caption from Florence-2
            detailed_info: Detailed caption information
            source: Source of the image
            metadata: Optional additional metadata

        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()

        # Generate document ID
        doc_id = Document.generate_doc_id()

        # Save image file
        file_path = self._save_file(image_bytes, doc_id, Path(filename).suffix)

        # Use GLM API to generate rich description and tags (pass filename for fallback)
        ai_description = await self.chatbot.generate_image_description(caption, filename)
        detailed_description = ai_description["detailed_description"]
        ai_summary = ai_description["summary"]
        ai_tags = ai_description["tags"]

        # Create enhanced content for embedding (description + tags)
        tag_keywords = ", ".join(ai_tags) if ai_tags else ""
        enhanced_content = f"{detailed_description}\n\nKeywords: {tag_keywords}".strip()

        # Create chunks with AI-enhanced content
        chunk_id = Chunk.generate_chunk_id()
        chunk = {
            "chunk_id": chunk_id,
            "content": enhanced_content,
            "chunk_number": 0,
            "token_count": len(enhanced_content) // 4,
        }

        # Generate embedding from enhanced content
        embedding = self.embedding_model.encode_single(enhanced_content)

        # Get image dimensions
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size

        # Combine AI tags with any detected tags
        all_tags = [(tag, 0.9) for tag in ai_tags]  # AI tags get high confidence
        for tag in detailed_info.get("tags", []):
            all_tags.append((tag, 0.7))  # Detected tags get slightly lower confidence

        # Save to database
        async with get_db_session() as session:
            # Create document record with AI-generated summary
            doc = await DocumentRepository.create(
                session,
                doc_id=doc_id,
                filename=filename,
                content_type="image",
                file_path=str(file_path),
                file_size=len(image_bytes),
                caption=ai_summary,  # Use AI-generated summary as caption
                summary=detailed_description,  # Store detailed description
                width=width,
                height=height,
                source=source,
            )

            # Create chunk record
            await ChunkRepository.create(
                session,
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=enhanced_content,
                faiss_index=self.vector_store.image_store.next_index,
                chunk_number=0,
                token_count=chunk["token_count"],
            )

            # Add AI-generated tags
            for tag_name, score in all_tags:
                await TagRepository.add_tag_to_document(
                    session, doc_id, tag_name, score, "ai"
                )

        # Add embedding to FAISS
        chunk_ids = [chunk_id]
        self.vector_store.add_image_embeddings(np.array([embedding]), chunk_ids)
        self.vector_store.save_all()

        processing_time = time.time() - start_time

        # Track with MLflow
        self.mlflow_tracker.track_ingestion(
            doc_type="image",
            doc_count=1,
            chunk_count=1,
            total_tokens=chunk["token_count"],
            processing_time=processing_time,
            filename=filename
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "ingested",
            "caption": ai_summary,
            "summary": detailed_description,
            "tags": [{"name": t[0], "score": t[1]} for t in all_tags],
            "created_at": datetime.utcnow().isoformat(),
        }

    def _save_file(self, content: bytes, doc_id: str, suffix: str) -> Path:
        """Save file content to disk"""
        settings.FILES_DIR.mkdir(parents=True, exist_ok=True)
        file_path = settings.FILES_DIR / f"{doc_id}{suffix}"
        with open(file_path, 'wb') as f:
            f.write(content)
        return file_path

    # ========================================================================
    # Retrieval Pipeline
    # ========================================================================

    async def retrieve(
        self,
        query: str,
        k: int = None,
        threshold: float = None,
        content_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity threshold
            content_type: Filter by content type ('text', 'image', or None for both)

        Returns:
            List of retrieved documents with scores
        """
        start_time = time.time()

        k = k or settings.TOP_K_RESULTS
        threshold = threshold or settings.SIMILARITY_THRESHOLD

        # Encode query
        query_embedding = self.embedding_model.encode_single(query)

        # Search
        if content_type == "text":
            results = self.vector_store.search_text(query_embedding, k, threshold)
            results_by_chunk_id = {r[2]: r for r in results}  # (index, score, chunk_id)
        elif content_type == "image":
            results = self.vector_store.search_images(query_embedding, k, threshold)
            results_by_chunk_id = {r[2]: r for r in results}  # (index, score, chunk_id)
        else:
            results = self.vector_store.search_combined(query_embedding, k, threshold)
            # search_combined returns (index, score, chunk_id, source)
            results_by_chunk_id = {r[2]: r for r in results}

        # Get chunk details from database
        faiss_indices = [r[0] for r in results]

        async with get_db_session() as session:
            chunks = await ChunkRepository.get_by_faiss_indices(session, faiss_indices)

        # Build results with documents - DEDUPLICATE by doc_id
        retrieved = []
        seen_doc_ids = {}  # doc_id -> {best_score, best_chunk, all_chunk_ids}

        for chunk in chunks:
            score_data = results_by_chunk_id.get(chunk.chunk_id)
            if not score_data:
                continue

            # Handle both 3-tuple and 4-tuple results
            faiss_idx = score_data[0]
            score = score_data[1]
            chunk_id = score_data[2]

            # Get document (fetch from DB or cache)
            async with get_db_session() as session:
                doc = await DocumentRepository.get_by_id(session, chunk.doc_id)

            if not doc:
                continue

            # Track best chunk per document
            if chunk.doc_id not in seen_doc_ids or score > seen_doc_ids[chunk.doc_id]["best_score"]:
                seen_doc_ids[chunk.doc_id] = {
                    "best_score": score,
                    "best_chunk": chunk,
                    "all_chunk_ids": seen_doc_ids.get(chunk.doc_id, {}).get("all_chunk_ids", [])
                }
            # Always add chunk_id to the list
            if chunk.doc_id not in seen_doc_ids:
                seen_doc_ids[chunk.doc_id] = {"best_score": score, "best_chunk": chunk, "all_chunk_ids": []}
            seen_doc_ids[chunk.doc_id]["all_chunk_ids"].append(chunk_id)

        # Build deduplicated results
        for doc_id, data in seen_doc_ids.items():
            best_chunk = data["best_chunk"]
            score = data["best_score"]
            all_chunk_ids = data["all_chunk_ids"]

            # Get document
            async with get_db_session() as session:
                doc = await DocumentRepository.get_by_id(session, doc_id)

            if not doc:
                continue

            retrieved.append({
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "chunk_id": best_chunk.chunk_id,
                "content": best_chunk.content,
                "summary": doc.summary or doc.caption,
                "score": score,
                "matched_chunks": len(all_chunk_ids),
                "metadata": {
                    "file_path": str(doc.file_path),
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                }
            })

        # Track with MLflow
        retrieval_time = time.time() - start_time
        avg_score = sum(r["score"] for r in retrieved) / len(retrieved) if retrieved else 0
        self.mlflow_tracker.track_retrieval(
            query=query,
            retrieved_count=len(retrieved),
            avg_score=avg_score,
            retrieval_time=retrieval_time,
            content_type=content_type or "all"
        )

        return retrieved

    async def retrieve_all_chunks(
        self,
        query: str,
        k: int = None,
        threshold: float = None,
        content_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ALL relevant chunks for a query (no deduplication by document).
        Use this for RAG context generation to get all matched content.

        Args:
            query: Search query
            k: Number of results to return
            threshold: Similarity threshold
            content_type: Filter by content type ('text', 'image', or None for both)

        Returns:
            List of ALL retrieved chunks with scores (may include multiple chunks from same document)
        """
        start_time = time.time()

        k = k or settings.TOP_K_RESULTS
        threshold = threshold or settings.SIMILARITY_THRESHOLD

        # Encode query
        query_embedding = self.embedding_model.encode_single(query)

        # Search
        if content_type == "text":
            results = self.vector_store.search_text(query_embedding, k, threshold)
            results_by_chunk_id = {r[2]: r for r in results}  # (index, score, chunk_id)
        elif content_type == "image":
            results = self.vector_store.search_images(query_embedding, k, threshold)
            results_by_chunk_id = {r[2]: r for r in results}  # (index, score, chunk_id)
        else:
            results = self.vector_store.search_combined(query_embedding, k, threshold)
            # search_combined returns (index, score, chunk_id, source)
            results_by_chunk_id = {r[2]: r for r in results}

        # Get chunk details from database
        faiss_indices = [r[0] for r in results]

        async with get_db_session() as session:
            chunks = await ChunkRepository.get_by_faiss_indices(session, faiss_indices)

        # Return ALL chunks with their documents (NO deduplication)
        retrieved_chunks = []

        for chunk in chunks:
            score_data = results_by_chunk_id.get(chunk.chunk_id)
            if not score_data:
                continue

            # Handle both 3-tuple and 4-tuple results
            score = score_data[1]

            # Get document
            async with get_db_session() as session:
                doc = await DocumentRepository.get_by_id(session, chunk.doc_id)

            if not doc:
                continue

            retrieved_chunks.append({
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "chunk_id": chunk.chunk_id,
                "chunk_number": chunk.chunk_number,
                "content": chunk.content,
                "summary": doc.summary or doc.caption,
                "score": score,
                "metadata": {
                    "file_path": str(doc.file_path),
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                }
            })

        # Sort by score descending
        retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Track with MLflow
        retrieval_time = time.time() - start_time
        avg_score = sum(r["score"] for r in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0
        self.mlflow_tracker.track_retrieval(
            query=query,
            retrieved_count=len(retrieved_chunks),
            avg_score=avg_score,
            retrieval_time=retrieval_time,
            content_type=content_type or "all"
        )

        return retrieved_chunks

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get full document details"""
        async with get_db_session() as session:
            doc = await DocumentRepository.get_by_id(session, doc_id)
            if not doc:
                return None

            chunks = await ChunkRepository.get_by_document(session, doc_id)
            tags = await TagRepository.get_by_document(session, doc_id)

            return {
                **doc.to_dict(),
                "chunks": [c.to_dict() for c in chunks],
                "tags": [t.to_dict() for t in tags],
            }

    async def list_documents(
        self,
        content_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all documents"""
        async with get_db_session() as session:
            docs = await DocumentRepository.list_all(session, content_type, limit, offset)
            return [doc.to_dict() for doc in docs]

    async def list_tags(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all tags"""
        async with get_db_session() as session:
            tags = await TagRepository.get_popular(session, limit)
            return [tag.to_dict() for tag in tags]

    # ========================================================================
    # RAG Generation Pipeline
    # ========================================================================

    async def generate_rag_context(
        self,
        query: str,
        k: int = None,
        max_context_length: int = None
    ) -> str:
        """
        Generate context string from retrieved documents

        Args:
            query: User query
            k: Number of chunks to retrieve
            max_context_length: Maximum context length in characters

        Returns:
            Context string for RAG
        """
        k = k or settings.TOP_K_RESULTS
        max_context_length = max_context_length or settings.RAG_CONTEXT_LENGTH

        # Retrieve ALL relevant chunks (no deduplication)
        chunks = await self.retrieve_all_chunks(query, k=k)

        if not chunks:
            return "No relevant information found in the knowledge base."

        # Build context from ALL chunks
        context_parts = []
        current_length = 0

        for i, chunk in enumerate(chunks):
            part = f"[Source {i+1}] {chunk['filename']} (chunk {chunk['chunk_number']})\n{chunk['content']}\n"
            if current_length + len(part) > max_context_length:
                # Truncate last part
                remaining = max_context_length - current_length
                if remaining > 50:
                    context_parts.append(part[:remaining] + "...")
                break
            context_parts.append(part)
            current_length += len(part)

        return "\n".join(context_parts)

    def build_rag_prompt(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Build RAG prompt with context and conversation history

        Args:
            query: User query
            context: Retrieved context
            conversation_history: Optional conversation history

        Returns:
            Formatted prompt for LLM
        """
        prompt_parts = [
            "You are a helpful assistant that answers questions based on the "
            "provided context from a knowledge base. Use the context to inform "
            "your answers, but don't mention the context explicitly in your response.",
            "",
            "Context:",
            context,
            "",
        ]

        # Add conversation history if provided
        if conversation_history:
            prompt_parts.append("Conversation History:")
            for msg in conversation_history[-4:]:  # Last 4 messages
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("")

        prompt_parts.extend([
            f"User: {query}",
            "Assistant:",
        ])

        return "\n".join(prompt_parts)


# ============================================================================
# Global RAG Pipeline Instance
# ============================================================================

_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def init_rag_pipeline():
    """Initialize the RAG pipeline"""
    global _rag_pipeline
    _rag_pipeline = RAGPipeline()
    return _rag_pipeline
