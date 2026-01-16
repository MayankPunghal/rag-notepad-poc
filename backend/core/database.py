"""
Database module for RAG Brain
SQLite database for storing metadata, documents, and embeddings mapping
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, Boolean,
    ForeignKey, JSON, Index
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func

from config.settings import settings

# SQLAlchemy Base
Base = declarative_base()


# ============================================================================
# Database Models
# ============================================================================

class Document(Base):
    """
    Document model for storing ingested documents (text and images)
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    content_type = Column(String(50), nullable=False)  # 'text', 'image'
    file_path = Column(String(1024), nullable=False)
    file_size = Column(Integer, default=0)

    # Text-specific fields
    raw_text = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)

    # Image-specific fields
    caption = Column(Text, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    # Metadata
    source = Column(String(100), default="upload")  # upload, url, etc.
    language = Column(String(10), default="en")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    tags = relationship("DocumentTag", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary"""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "raw_text": self.raw_text,
            "summary": self.summary,
            "caption": self.caption,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "language": self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Chunk(Base):
    """
    Chunk model for storing text chunks and their embedding indices
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(64), unique=True, nullable=False, index=True)
    doc_id = Column(String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)

    # Chunk content
    content = Column(Text, nullable=False)

    # FAISS index
    faiss_index = Column(Integer, nullable=False, index=True)  # Index in FAISS vector

    # Chunk metadata
    chunk_number = Column(Integer, default=0)
    token_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    # Indexes
    __table_args__ = (
        Index("idx_chunk_doc", "doc_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "faiss_index": self.faiss_index,
            "chunk_number": self.chunk_number,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Tag(Base):
    """
    Tag model for storing auto-generated and manual tags
    """
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tag_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), default="general")  # general, topic, entity, etc.
    count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    documents = relationship("DocumentTag", back_populates="tag", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert tag to dictionary"""
        return {
            "tag_id": self.tag_id,
            "name": self.name,
            "category": self.category,
            "count": self.count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DocumentTag(Base):
    """
    Junction table for Document-Tag many-to-many relationship
    """
    __tablename__ = "document_tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    tag_id = Column(String(64), ForeignKey("tags.tag_id", ondelete="CASCADE"), nullable=False)

    # Tag score (confidence for auto-generated tags)
    score = Column(Float, default=1.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="tags")
    tag = relationship("Tag", back_populates="documents")

    # Indexes
    __table_args__ = (
        Index("idx_doc_tag_doc", "doc_id"),
        Index("idx_doc_tag_tag", "tag_id"),
    )


class Conversation(Base):
    """
    Conversation model for storing chat sessions
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(255), default="New Conversation")
    model = Column(String(100), default="glm-4")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary"""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "model": self.model,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Message(Base):
    """
    Message model for storing chat messages in conversations
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(64), unique=True, nullable=False, index=True)
    conversation_id = Column(String(64), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)

    # Message content
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)

    # Context used for generation
    context_docs = Column(JSON, nullable=True)  # List of doc_ids used as context

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    # Indexes
    __table_args__ = (
        Index("idx_message_conv", "conversation_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "context_docs": self.context_docs,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================================
# Database Engine
# ============================================================================

# Sync engine for migrations and initial setup
sync_engine = None
sync_session_factory = None

# Async engine for application use
async_engine = None
async_session_factory = None


def init_database():
    """Initialize database connection and create tables"""
    global sync_engine, sync_session_factory

    # Ensure data directory exists
    settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create sync engine
    db_url = f"sqlite:///{settings.SQLITE_DB_PATH}"
    sync_engine = create_engine(db_url, echo=settings.DEBUG, connect_args={"check_same_thread": False})
    sync_session_factory = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)

    # Create all tables
    Base.metadata.create_all(sync_engine)


async def init_async_database():
    """Initialize async database connection"""
    global async_engine, async_session_factory

    # Ensure data directory exists
    settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create async engine
    db_url = f"sqlite+aiosqlite:///{settings.SQLITE_DB_PATH}"
    async_engine = create_async_engine(db_url, echo=settings.DEBUG)
    async_session_factory = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create all tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_db_session():
    """Get async database session context manager"""
    if async_session_factory is None:
        await init_async_database()

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_sync_session():
    """Get sync database session (for migrations, tests)"""
    if sync_session_factory is None:
        init_database()

    return sync_session_factory()


# ============================================================================
# Repository Classes
# ============================================================================

class DocumentRepository:
    """Repository for Document operations"""

    @staticmethod
    async def create(session: AsyncSession, **kwargs) -> Document:
        """Create a new document"""
        doc = Document(**kwargs)
        session.add(doc)
        await session.flush()
        await session.refresh(doc)
        return doc

    @staticmethod
    async def get_by_id(session: AsyncSession, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        result = await session.execute(
            select(Document).where(Document.doc_id == doc_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_all(
        session: AsyncSession,
        content_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Document]:
        """List all documents with optional filtering"""
        query = select(Document)
        if content_type:
            query = query.where(Document.content_type == content_type)
        query = query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def search(
        session: AsyncSession,
        query: str,
        limit: int = 20
    ) -> List[Document]:
        """Full-text search in documents"""
        # Search in text content and summaries
        result = await session.execute(
            select(Document).where(
                (Document.raw_text.ilike(f"%{query}%")) |
                (Document.summary.ilike(f"%{query}%")) |
                (Document.caption.ilike(f"%{query}%"))
            ).limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update(session: AsyncSession, doc_id: str, **kwargs) -> Optional[Document]:
        """Update document"""
        doc = await DocumentRepository.get_by_id(session, doc_id)
        if doc:
            for key, value in kwargs.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)
            doc.updated_at = datetime.utcnow()
            await session.flush()
            await session.refresh(doc)
        return doc

    @staticmethod
    async def delete(session: AsyncSession, doc_id: str) -> bool:
        """Delete document"""
        doc = await DocumentRepository.get_by_id(session, doc_id)
        if doc:
            await session.delete(doc)
            return True
        return False


class ChunkRepository:
    """Repository for Chunk operations"""

    @staticmethod
    async def create(session: AsyncSession, **kwargs) -> Chunk:
        """Create a new chunk"""
        chunk = Chunk(**kwargs)
        session.add(chunk)
        await session.flush()
        await session.refresh(chunk)
        return chunk

    @staticmethod
    async def get_by_faiss_indices(
        session: AsyncSession,
        indices: List[int]
    ) -> List[Chunk]:
        """Get chunks by their FAISS indices"""
        if not indices:
            return []
        result = await session.execute(
            select(Chunk).where(Chunk.faiss_index.in_(indices))
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_by_document(session: AsyncSession, doc_id: str) -> List[Chunk]:
        """Get all chunks for a document"""
        result = await session.execute(
            select(Chunk).where(
                Chunk.doc_id == doc_id
            ).order_by(Chunk.chunk_number)
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete_by_document(session: AsyncSession, doc_id: str) -> int:
        """Delete all chunks for a document"""
        chunks = await ChunkRepository.get_by_document(session, doc_id)
        count = len(chunks)
        for chunk in chunks:
            await session.delete(chunk)
        return count


class TagRepository:
    """Repository for Tag operations"""

    @staticmethod
    async def get_or_create(session: AsyncSession, name: str, category: str = "general") -> Tag:
        """Get existing tag or create new one"""
        result = await session.execute(
            select(Tag).where(Tag.name == name)
        )
        tag = result.scalar_one_or_none()
        if not tag:
            tag = Tag(tag_id=Tag.generate_tag_id(), name=name, category=category)
            session.add(tag)
            await session.flush()
            await session.refresh(tag)
        return tag

    @staticmethod
    async def add_tag_to_document(
        session: AsyncSession,
        doc_id: str,
        tag_name: str,
        score: float = 1.0,
        category: str = "general"
    ) -> DocumentTag:
        """Add tag to document"""
        tag = await TagRepository.get_or_create(session, tag_name, category)

        # Check if already tagged
        result = await session.execute(
            select(DocumentTag).where(
                DocumentTag.doc_id == doc_id,
                DocumentTag.tag_id == tag.tag_id
            )
        )
        doc_tag = result.scalar_one_or_none()

        if not doc_tag:
            doc_tag = DocumentTag(
                doc_id=doc_id,
                tag_id=tag.tag_id,
                score=score
            )
            session.add(doc_tag)
            tag.count += 1
            await session.flush()
            await session.refresh(doc_tag)

        return doc_tag

    @staticmethod
    async def get_by_document(session: AsyncSession, doc_id: str) -> List[Tag]:
        """Get all tags for a document"""
        result = await session.execute(
            select(Tag).join(DocumentTag).where(
                DocumentTag.doc_id == doc_id
            ).order_by(Tag.count.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_popular(session: AsyncSession, limit: int = 20) -> List[Tag]:
        """Get most popular tags"""
        result = await session.execute(
            select(Tag).order_by(Tag.count.desc()).limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete_by_document(session: AsyncSession, doc_id: str) -> int:
        """Delete all tag associations for a document and remove orphaned tags"""
        # Get all DocumentTag associations for this document
        result = await session.execute(
            select(DocumentTag).where(DocumentTag.doc_id == doc_id)
        )
        doc_tags = result.scalars().all()
        count = len(doc_tags)

        # Track which tags to potentially delete
        tags_to_check = set()

        # Delete associations and decrement tag counts
        for doc_tag in doc_tags:
            tags_to_check.add(doc_tag.tag_id)

            # Decrement tag count
            tag = await session.execute(
                select(Tag).where(Tag.tag_id == doc_tag.tag_id)
            )
            tag_obj = tag.scalar_one_or_none()
            if tag_obj:
                tag_obj.count = max(0, tag_obj.count - 1)

            # Delete the association
            await session.delete(doc_tag)

        # Delete tags that now have count 0 (orphaned tags)
        for tag_id in tags_to_check:
            tag = await session.execute(
                select(Tag).where(Tag.tag_id == tag_id)
            )
            tag_obj = tag.scalar_one_or_none()
            if tag_obj and tag_obj.count == 0:
                await session.delete(tag_obj)

        return count


class ConversationRepository:
    """Repository for Conversation operations"""

    @staticmethod
    async def create(session: AsyncSession, title: str = "New Conversation") -> Conversation:
        """Create a new conversation"""
        conv = Conversation(
            conversation_id=Conversation.generate_conv_id(),
            title=title
        )
        session.add(conv)
        await session.flush()
        await session.refresh(conv)
        return conv

    @staticmethod
    async def get_by_id(session: AsyncSession, conv_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        result = await session.execute(
            select(Conversation).where(Conversation.conversation_id == conv_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_all(session: AsyncSession, limit: int = 50) -> List[Conversation]:
        """List all conversations"""
        result = await session.execute(
            select(Conversation).order_by(
                Conversation.updated_at.desc()
            ).limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def add_message(
        session: AsyncSession,
        conv_id: str,
        role: str,
        content: str,
        context_docs: Optional[List[str]] = None
    ) -> Message:
        """Add message to conversation"""
        msg = Message(
            message_id=Message.generate_msg_id(),
            conversation_id=conv_id,
            role=role,
            content=content,
            context_docs=context_docs
        )
        session.add(msg)

        # Update conversation timestamp
        conv = await ConversationRepository.get_by_id(session, conv_id)
        if conv:
            conv.updated_at = datetime.utcnow()

        await session.flush()
        await session.refresh(msg)
        return msg

    @staticmethod
    async def get_messages(session: AsyncSession, conv_id: str) -> List[Message]:
        """Get all messages in a conversation"""
        result = await session.execute(
            select(Message).where(
                Message.conversation_id == conv_id
            ).order_by(Message.created_at.asc())
        )
        return list(result.scalars().all())


# Add helper methods to models
from sqlalchemy import select

def generate_uuid_like_id(prefix: str = "") -> str:
    """Generate a UUID-like ID string"""
    import uuid
    uid = str(uuid.uuid4()).replace("-", "")[:16]
    return f"{prefix}{uid}" if prefix else uid


Document.generate_doc_id = classmethod(lambda cls: generate_uuid_like_id("doc_"))
Chunk.generate_chunk_id = classmethod(lambda cls: generate_uuid_like_id("chk_"))
Tag.generate_tag_id = classmethod(lambda cls: generate_uuid_like_id("tag_"))
Conversation.generate_conv_id = classmethod(lambda cls: generate_uuid_like_id("conv_"))
Message.generate_msg_id = classmethod(lambda cls: generate_uuid_like_id("msg_"))
