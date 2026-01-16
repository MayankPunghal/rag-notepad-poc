"""
RAG Brain - Main FastAPI Application
Multimodal RAG system with local embeddings and GLM chatbot
"""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any
import io

import faiss
from fastapi import (
    FastAPI, UploadFile, File, Form, HTTPException, Query,
    BackgroundTasks, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from config.settings import settings
from config.logging_config import setup_logging
from backend.core.database import init_async_database, get_db_session, Document, Chunk, DocumentTag, Tag
from backend.core.vector_store import init_vector_store, get_vector_store
from backend.services.rag_pipeline import get_rag_pipeline
from backend.services.glm_chatbot import get_chatbot, init_chatbot
from backend.services.mlflow_tracker import get_mlflow_tracker, init_mlflow
from backend.models.embeddings import (
    get_image_captioning_model, ImageUtils
)


# ============================================================================
# Request/Response Models
# ============================================================================

class IngestTextRequest(BaseModel):
    """Request model for text ingestion"""
    text: str = Field(..., description="Text content to ingest")
    filename: str = Field(default="document.txt", description="Original filename")
    source: str = Field(default="upload", description="Source of the document")


class IngestTextResponse(BaseModel):
    """Response model for text ingestion"""
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    summary: str
    tags: List[Dict[str, Any]]
    created_at: str


class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query")
    k: int = Field(default=5, description="Number of results", ge=1, le=50)
    threshold: float = Field(default=0.5, description="Similarity threshold", ge=0, le=1)
    content_type: Optional[str] = Field(
        default=None,
        description="Filter by content type (text, image, or None for both)"
    )


class SearchResult(BaseModel):
    """Model for a single search result"""
    doc_id: str
    filename: str
    content_type: str
    chunk_id: str
    content: str
    summary: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for semantic search"""
    query: str
    results: List[SearchResult]
    total_count: int
    retrieval_time: float


class ChatRequest(BaseModel):
    """Request model for chat"""
    query: str = Field(..., description="User query")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    use_rag: bool = Field(default=True, description="Whether to use RAG retrieval")
    k: int = Field(default=10, description="Number of documents to retrieve for RAG", ge=1, le=50)
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response model for chat"""
    query: str
    response: str
    context_docs: List[str]
    context_used: bool
    generation_time: float
    total_time: float
    model: str
    mock_mode: bool = False
    original_error: Optional[str] = None


class ConversationCreate(BaseModel):
    """Request model for creating a conversation"""
    title: str = Field(default="New Conversation", description="Conversation title")


class Message(BaseModel):
    """Message model"""
    role: str
    content: str
    created_at: Optional[str] = None


class Conversation(BaseModel):
    """Conversation model"""
    conversation_id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    messages: List[Message] = []


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    vector_store_stats: Optional[Dict[str, Any]] = None
    mlflow_stats: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    """Statistics response"""
    document_count: int
    text_count: int
    image_count: int
    chunk_count: int
    tag_count: int
    conversation_count: int
    vector_store_stats: Dict[str, Any]


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Initialize logging
    logger = setup_logging()

    # Startup
    print("Starting RAG Brain...")
    logger.info("Starting RAG Brain...")

    # Initialize database
    print("Initializing database...")
    logger.info("Initializing database...")
    await init_async_database()

    # Initialize vector store
    print("Initializing vector store...")
    logger.info("Initializing vector store...")
    init_vector_store()

    # Initialize MLflow tracking
    print("Initializing MLflow tracking...")
    logger.info("Initializing MLflow tracking...")
    init_mlflow()

    # Initialize chatbot
    print("Initializing chatbot...")
    logger.info("Initializing chatbot...")
    init_chatbot()

    # Get initial stats
    vector_stats = get_vector_store().get_all_stats()
    print(f"Vector store stats: {vector_stats}")
    logger.info(f"Vector store stats: {vector_stats}")

    print("RAG Brain started successfully!")
    logger.info("RAG Brain started successfully!")

    yield

    # Shutdown
    print("Shutting down RAG Brain...")
    logger.info("Shutting down RAG Brain...")
    # Save vector store
    get_vector_store().save_all()
    print("RAG Brain shutdown complete.")
    logger.info("RAG Brain shutdown complete.")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Local multimodal RAG system with semantic search and GLM chatbot",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/js", StaticFiles(directory=str(frontend_path / "js")), name="js")

# ============================================================================
# Frontend Routes
# ============================================================================

@app.get("/", tags=["Frontend"])
async def frontend_index():
    """Serve the frontend index.html"""
    frontend_file = frontend_path / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    return {"message": "Frontend not found. Please build the frontend first."}


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    vector_store = get_vector_store()
    mlflow_tracker = get_mlflow_tracker()

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        models_loaded=True,
        vector_store_stats=vector_store.get_all_stats(),
        mlflow_stats=mlflow_tracker.get_stats(),
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_statistics():
    """Get system statistics"""
    from backend.core.database import (
        DocumentRepository, ChunkRepository, TagRepository, ConversationRepository
    )

    async with get_db_session() as session:
        all_docs = await DocumentRepository.list_all(session, limit=100000)
        text_docs = await DocumentRepository.list_all(session, content_type="text", limit=100000)
        image_docs = await DocumentRepository.list_all(session, content_type="image", limit=100000)
        tags = await TagRepository.get_popular(session, limit=100000)
        convs = await ConversationRepository.list_all(session, limit=100000)

    vector_store = get_vector_store()

    return StatsResponse(
        document_count=len(all_docs),
        text_count=len(text_docs),
        image_count=len(image_docs),
        chunk_count=vector_store.text_store.get_stats()["total_vectors"] +
                   vector_store.image_store.get_stats()["total_vectors"],
        tag_count=len(tags),
        conversation_count=len(convs),
        vector_store_stats=vector_store.get_all_stats(),
    )


@app.get("/api/v1/mlflow/stats", tags=["MLflow"])
async def get_mlflow_stats():
    """Get MLflow tracking statistics and experiments"""
    from backend.services.mlflow_tracker import get_mlflow_tracker

    tracker = get_mlflow_tracker()

    stats = tracker.get_stats()
    experiments = tracker.tracker.list_experiments() if stats.get("initialized") else []

    # Get recent runs from the default experiment
    recent_runs = []
    if stats.get("initialized") and experiments:
        for exp in experiments[:3]:  # Last 3 experiments
            runs = tracker.tracker.list_runs(exp["id"])
            recent_runs.extend(runs[:5])  # Last 5 runs per experiment

    return {
        "mlflow_available": stats.get("mlflow_available", False),
        "initialized": stats.get("initialized", False),
        "tracking_uri": stats.get("tracking_uri", ""),
        "experiment_name": stats.get("experiment_name", ""),
        "experiments": experiments,
        "recent_runs": recent_runs[:10],  # Last 10 runs total
    }


@app.get("/api/v1/mlflow/experiments", tags=["MLflow"])
async def list_mlflow_experiments():
    """List all MLflow experiments"""
    from backend.services.mlflow_tracker import get_mlflow_tracker

    tracker = get_mlflow_tracker()
    experiments = tracker.tracker.list_experiments()

    return {
        "experiments": experiments,
        "count": len(experiments),
    }


@app.get("/api/v1/mlflow/runs", tags=["MLflow"])
async def list_mlflow_runs(limit: int = 20):
    """List recent MLflow runs"""
    from backend.services.mlflow_tracker import get_mlflow_tracker

    tracker = get_mlflow_tracker()
    experiments = tracker.tracker.list_experiments()

    all_runs = []
    for exp in experiments:
        runs = tracker.tracker.list_runs(exp["id"])
        all_runs.extend(runs)

    # Sort by start time (most recent first)
    all_runs.sort(key=lambda x: x.get("start_time", 0), reverse=True)

    return {
        "runs": all_runs[:limit],
        "count": len(all_runs),
    }


@app.get("/api/v1/logs", tags=["System"])
async def get_logs(
    lines: int = Query(default=100, ge=1, le=1000, description="Number of recent log lines"),
    log_type: str = Query(default="rag_brain", description="Log file name (rag_brain or errors)")
):
    """Get recent application logs from the log file"""
    import os

    log_file = settings.LOGS_DIR / f"{log_type}.log"

    if not log_file.exists():
        return {
            "logs": "Log file not found. Logging may not be initialized yet.",
            "log_file": str(log_file),
            "exists": False
        }

    try:
        # Read last N lines from log file
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Read all lines and get the last N
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "logs": "".join(recent_lines),
            "log_file": str(log_file),
            "exists": True,
            "line_count": len(recent_lines),
            "total_lines": len(all_lines)
        }
    except Exception as e:
        return {
            "logs": f"Error reading log file: {str(e)}",
            "log_file": str(log_file),
            "exists": True,
            "error": str(e)
        }


# ============================================================================
# Ingestion Endpoints
# ============================================================================

@app.post("/api/v1/ingest/text", response_model=IngestTextResponse, tags=["Ingestion"])
async def ingest_text(request: IngestTextRequest):
    """Ingest a text document"""
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text content is too short")

    start_time = time.time()

    try:
        rag_pipeline = get_rag_pipeline()
        mlflow_tracker = get_mlflow_tracker()

        result = await rag_pipeline.ingest_text(
            text=request.text,
            filename=request.filename,
            source=request.source,
        )

        processing_time = time.time() - start_time

        # Track in MLflow
        mlflow_tracker.track_ingestion(
            doc_type="text",
            doc_count=1,
            chunk_count=result["chunk_count"],
            total_tokens=len(request.text.split()),
            processing_time=processing_time,
            filename=request.filename,
        )

        return IngestTextResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/v1/ingest/file", tags=["Ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Ingest a file (text or image)"""
    # Validate file size
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE // (1024*1024)}MB"
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_TEXT_EXTENSIONS + settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {settings.ALLOWED_TEXT_EXTENSIONS + settings.ALLOWED_IMAGE_EXTENSIONS}"
        )

    start_time = time.time()

    try:
        rag_pipeline = get_rag_pipeline()
        mlflow_tracker = get_mlflow_tracker()

        # Check if image
        if file_ext in settings.ALLOWED_IMAGE_EXTENSIONS:
            # Process image
            captioning_model = get_image_captioning_model()
            image_utils = ImageUtils()

            # Load image
            image = image_utils.load_image(content)
            image_info = image_utils.get_image_info(image)

            # Generate caption
            caption_info = captioning_model.generate_detailed_caption(image)

            # Ingest
            result = await rag_pipeline.ingest_image(
                image_bytes=content,
                filename=file.filename,
                caption=caption_info["caption"],
                detailed_info=caption_info,
                source="upload",
            )

            processing_time = time.time() - start_time

            mlflow_tracker.track_ingestion(
                doc_type="image",
                doc_count=1,
                chunk_count=1,
                total_tokens=len(caption_info["caption"].split()),
                processing_time=processing_time,
                filename=file.filename,
            )

            return {
                "doc_id": result["doc_id"],
                "filename": result["filename"],
                "content_type": "image",
                "status": result["status"],
                "caption": result["caption"],
                "tags": result["tags"],
                "image_info": image_info,
                "created_at": result["created_at"],
            }

        else:
            # Process text file
            from backend.services.text_processor import FileReader

            # Save and read file
            settings.FILES_DIR.mkdir(parents=True, exist_ok=True)
            file_path = settings.FILES_DIR / file.filename
            with open(file_path, 'wb') as f:
                f.write(content)

            # Extract text
            text, content_type = FileReader.read_file(file_path)

            if not text or len(text.strip()) < 10:
                raise HTTPException(status_code=400, detail="Could not extract text from file")

            # Ingest
            result = await rag_pipeline.ingest_text(
                text=text,
                filename=file.filename,
                source="file",
            )

            processing_time = time.time() - start_time

            mlflow_tracker.track_ingestion(
                doc_type="text",
                doc_count=1,
                chunk_count=result["chunk_count"],
                total_tokens=len(text.split()),
                processing_time=processing_time,
                filename=file.filename,
            )

            return IngestTextResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


# ============================================================================
# Search & Retrieval Endpoints
# ============================================================================

@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search(request: SearchRequest):
    """Perform semantic search"""
    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query is too short")

    start_time = time.time()

    try:
        rag_pipeline = get_rag_pipeline()
        mlflow_tracker = get_mlflow_tracker()

        # Retrieve documents
        retrieved = await rag_pipeline.retrieve(
            query=request.query,
            k=request.k,
            threshold=request.threshold,
            content_type=request.content_type,
        )

        retrieval_time = time.time() - start_time

        # Track in MLflow
        avg_score = sum(r["score"] for r in retrieved) / len(retrieved) if retrieved else 0
        mlflow_tracker.track_retrieval(
            query=request.query,
            retrieved_count=len(retrieved),
            avg_score=avg_score,
            retrieval_time=retrieval_time,
            content_type=request.content_type or "all",
        )

        return SearchResponse(
            query=request.query,
            results=[SearchResult(**r) for r in retrieved],
            total_count=len(retrieved),
            retrieval_time=retrieval_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/v1/documents", tags=["Documents"])
async def list_documents(
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """List all documents"""
    try:
        rag_pipeline = get_rag_pipeline()
        docs = await rag_pipeline.list_documents(content_type, limit, offset)
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/api/v1/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """Get document details"""
    try:
        rag_pipeline = get_rag_pipeline()
        doc = await rag_pipeline.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@app.get("/api/v1/files/{doc_id}", tags=["Documents"])
async def get_file(doc_id: str):
    """Get original file (for images)"""
    from backend.core.database import DocumentRepository, get_db_session

    async with get_db_session() as session:
        doc = await DocumentRepository.get_by_id(session, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = Path(doc.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        # Determine media type
        if doc.content_type == "image":
            media_type = f"image/{file_path.suffix[1:]}"
        else:
            media_type = "application/octet-stream"

        return FileResponse(file_path, media_type=media_type, filename=doc.filename)


@app.delete("/api/v1/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and all associated data (chunks, tags, vectors)"""
    from backend.core.database import DocumentRepository, ChunkRepository, TagRepository, get_db_session

    async with get_db_session() as session:
        doc = await DocumentRepository.get_by_id(session, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get chunks BEFORE deleting to remove from FAISS
        chunks = await ChunkRepository.get_by_document(session, doc_id)

        # Remove from FAISS vector store
        vector_store = get_vector_store()
        for chunk in chunks:
            # Delete from text store
            vector_store.text_store.delete(chunk.chunk_id)

        # Save FAISS changes
        vector_store.save_all()

        # Delete tags for this document
        await TagRepository.delete_by_document(session, doc_id)

        # Delete chunks
        await ChunkRepository.delete_by_document(session, doc_id)

        # Delete document
        await session.delete(doc)

        return {"status": "deleted", "doc_id": doc_id}


@app.post("/api/v1/cleanup", tags=["Documents"])
async def cleanup_orphaned_data():
    """Clean up orphaned tags and rebuild FAISS index from existing documents"""
    from backend.core.database import (
        DocumentRepository, ChunkRepository, TagRepository,
        DocumentTag, Tag, get_db_session
    )
    from backend.core.vector_store import get_vector_store
    from backend.services.text_processor import DocumentProcessor
    from backend.models.embeddings import get_text_embedding_model
    import numpy as np

    stats = {"tags_removed": 0, "faiss_rebuilt": False, "chunks_removed": 0}

    async with get_db_session() as session:
        # Clean up orphaned tags (tags with no documents and count > 0)
        result = await session.execute(
            select(Tag).where(Tag.count > 0)
        )
        tags_to_check = result.scalars().all()

        for tag in tags_to_check:
            # Check if any documents use this tag
            doc_tag_result = await session.execute(
                select(DocumentTag).where(DocumentTag.tag_id == tag.tag_id)
            )
            doc_tags = doc_tag_result.scalars().all()

            if len(doc_tags) == 0:
                # Tag has count but no actual document associations - delete it
                await session.delete(tag)
                stats["tags_removed"] += 1

        # Clean up orphaned DocumentTag entries (tags pointing to non-existent docs)
        # Get all doc_ids that exist
        docs_result = await session.execute(select(Document.doc_id))
        existing_doc_ids = set([row[0] for row in docs_result.fetchall()])

        # Get all DocumentTag entries
        all_doc_tags_result = await session.execute(select(DocumentTag))
        all_doc_tags = all_doc_tags_result.scalars().all()

        orphaned_doc_tags = []
        for dt in all_doc_tags:
            if dt.doc_id not in existing_doc_ids:
                orphaned_doc_tags.append(dt)

        for dt in orphaned_doc_tags:
            await session.delete(dt)
            # Also decrement the tag count
            tag = await session.execute(select(Tag).where(Tag.tag_id == dt.tag_id))
            tag_obj = tag.scalar_one_or_none()
            if tag_obj:
                tag_obj.count = max(0, tag_obj.count - 1)
                # Delete tag if count reaches 0
                if tag_obj.count == 0:
                    await session.delete(tag_obj)
            stats["tags_removed"] += 1

        # Clean up tags with count=0 (orphaned tags with no associations)
        zero_count_tags = await session.execute(
            select(Tag).where(Tag.count == 0)
        )
        for tag in zero_count_tags.scalars().all():
            await session.delete(tag)
            stats["tags_removed"] += 1

        # Clean up orphaned chunks (chunks pointing to non-existent docs)
        chunks_result = await session.execute(select(Chunk))
        all_chunks = chunks_result.scalars().all()

        orphaned_chunks = []
        for chunk in all_chunks:
            if chunk.doc_id not in existing_doc_ids:
                orphaned_chunks.append(chunk)

        for chunk in orphaned_chunks:
            await session.delete(chunk)
            stats["chunks_removed"] += 1

    # Rebuild FAISS index from existing documents
    try:
        vector_store = get_vector_store()
        embedding_model = get_text_embedding_model()
        doc_processor = DocumentProcessor()

        # Get all existing documents
        async with get_db_session() as session:
            docs_result = await session.execute(select(Document))
            documents = docs_result.scalars().all()

        if documents:
            # Clear the current index
            vector_store.text_store.index = faiss.IndexFlatIP(settings.TEXT_EMBEDDING_DIM)
            vector_store.text_store.id_to_index = {}
            vector_store.text_store.index_to_id = {}
            vector_store.text_store.next_index = 0

            # Re-index all documents
            async with get_db_session() as session:
                for doc in documents:
                    if doc.content_type == "text" and doc.raw_text:
                        # Process and re-embed
                        result = doc_processor.process_text_document(
                            doc.raw_text, doc.doc_id, doc.filename
                        )

                        # Get existing chunks to preserve their chunk_ids
                        chunks_result = await session.execute(
                            select(Chunk).where(Chunk.doc_id == doc.doc_id)
                        )
                        existing_chunks = chunks_result.scalars().all()

                        # Update chunk FAISS indices
                        for i, chunk_data in enumerate(result["chunks"]):
                            # Find matching existing chunk or use new
                            chunk_id = chunk_data["chunk_id"]
                            existing_chunk = None
                            for ec in existing_chunks:
                                if ec.chunk_id == chunk_id:
                                    existing_chunk = ec
                                    break

                            if existing_chunk:
                                # Update the chunk's FAISS index
                                existing_chunk.faiss_index = vector_store.text_store.next_index + i

                        # Add embeddings
                        chunk_ids = [c["chunk_id"] for c in result["chunks"]]
                        vector_store.add_text_embeddings(result["embeddings"], chunk_ids)

            vector_store.save_all()
            stats["faiss_rebuilt"] = True
    except Exception as e:
        stats["faiss_error"] = str(e)

    return {"status": "cleanup_complete", "stats": stats}


@app.get("/api/v1/tags", tags=["Documents"])
async def list_tags(limit: int = Query(50, ge=1, le=200)):
    """List all tags"""
    try:
        rag_pipeline = get_rag_pipeline()
        tags = await rag_pipeline.list_tags(limit)
        return {"tags": tags, "count": len(tags)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tags: {str(e)}")


# ============================================================================
# Chat Endpoints
# ============================================================================

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Send a chat message and get a response"""
    if not request.query or len(request.query.strip()) < 1:
        raise HTTPException(status_code=400, detail="Query is empty")

    # Check API key
    if not settings.GLM_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="GLM API key is not configured. Please set GLM_API_KEY environment variable."
        )

    try:
        chatbot = get_chatbot()

        # Create conversation if not provided
        conversation_id = request.conversation_id
        if not conversation_id:
            conv = await chatbot.create_conversation()
            conversation_id = conv["conversation_id"]

        response = await chatbot.chat(
            query=request.query,
            conversation_id=conversation_id,
            use_rag=request.use_rag,
            k=request.k,
        )

        return ChatResponse(**response)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/v1/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """Send a chat message and stream the response"""
    if not request.query or len(request.query.strip()) < 1:
        raise HTTPException(status_code=400, detail="Query is empty")

    # Check API key
    if not settings.GLM_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="GLM API key is not configured. Please set GLM_API_KEY environment variable."
        )

    async def generate():
        try:
            chatbot = get_chatbot()

            # Create conversation if not provided
            conversation_id = request.conversation_id
            if not conversation_id:
                conv = await chatbot.create_conversation()
                conversation_id = conv["conversation_id"]

            async for chunk in chatbot.chat_stream(
                query=request.query,
                conversation_id=conversation_id,
                use_rag=request.use_rag,
                k=request.k,
            ):
                yield f"data: {chunk.json()}\n\n"

        except Exception as e:
            yield f"data: {{'error': '{str(e)}', 'done': true}}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/v1/conversations", tags=["Chat"])
async def list_conversations(limit: int = Query(50, ge=1, le=200)):
    """List all conversations"""
    try:
        chatbot = get_chatbot()
        convs = await chatbot.list_conversations(limit)
        return {"conversations": convs, "count": len(convs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@app.get("/api/v1/conversations/{conversation_id}", response_model=Conversation, tags=["Chat"])
async def get_conversation(conversation_id: str):
    """Get conversation details"""
    try:
        chatbot = get_chatbot()
        conv = await chatbot.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


@app.post("/api/v1/conversations", response_model=Conversation, tags=["Chat"])
async def create_conversation(request: ConversationCreate = None):
    """Create a new conversation"""
    try:
        chatbot = get_chatbot()
        title = request.title if request else "New Conversation"
        conv = await chatbot.create_conversation(title)
        return conv
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


@app.delete("/api/v1/conversations/{conversation_id}", tags=["Chat"])
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        chatbot = get_chatbot()
        success = await chatbot.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "deleted", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
