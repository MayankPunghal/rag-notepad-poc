"""
Text Processing module for RAG Brain
Handles text chunking, summarization, and preprocessing
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np

from config.settings import settings
from backend.models.embeddings import get_text_embedding_model, get_auto_tagging_model


# ============================================================================
# Text Chunker
# ============================================================================

class TextChunker:
    """
    Split text into chunks for embedding and retrieval
    Handles different chunking strategies
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize text chunker

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_text(
        self,
        text: str,
        doc_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks

        Args:
            text: Input text
            doc_id: Document ID for chunk ID generation

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # Clean the text first
        text = self._clean_text(text)

        # Split into chunks
        chunks = self._split_by_size(text)

        # Post-process chunks
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(doc_id, i) if doc_id else f"chk_{i:04d}"

            # Estimate token count (rough approximation: 1 token ~ 4 chars)
            token_count = len(chunk) // 4

            chunk_dicts.append({
                "chunk_id": chunk_id,
                "content": chunk.strip(),
                "chunk_number": i,
                "token_count": token_count,
            })

        return chunk_dicts

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters except newlines
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def _split_by_size(self, text: str) -> List[str]:
        """Split text by character count while preserving sentence boundaries"""
        chunks = []

        # Find sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = ""
        for sentence in sentences:
            # If sentence alone exceeds chunk size, split it
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split long sentence by words
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                        temp_chunk += (" " if temp_chunk else "") + word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = word
                current_chunk = temp_chunk
            else:
                # Check if adding this sentence exceeds chunk size
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        # Add overlap to chunks (except first)
        if self.chunk_overlap > 0:
            chunks_with_overlap = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_chunk = chunks_with_overlap[-1]
                overlap_text = prev_chunk[-self.chunk_overlap:]
                chunks_with_overlap.append(overlap_text + " " + chunks[i])
            chunks = chunks_with_overlap

        return chunks

    def _generate_chunk_id(self, doc_id: str, chunk_number: int) -> str:
        """Generate a unique chunk ID"""
        content = f"{doc_id}_{chunk_number}_{datetime.utcnow().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def chunk_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents in batch

        Args:
            documents: List of document dictionaries with doc_id and text

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(
                doc.get("text", ""),
                doc.get("doc_id")
            )
            for chunk in chunks:
                chunk["doc_id"] = doc.get("doc_id")
                all_chunks.append(chunk)
        return all_chunks


# ============================================================================
# Text Summarizer
# ============================================================================

class TextSummarizer:
    """
    Local text summarization using extractive summarization
    CPU-optimized approach
    """

    def __init__(
        self,
        max_length: int = None,
        embedding_model = None
    ):
        """
        Initialize text summarizer

        Args:
            max_length: Maximum summary length in characters
            embedding_model: Text embedding model for similarity scoring
        """
        self.max_length = max_length or settings.MAX_SUMMARY_LENGTH
        self.embedding_model = embedding_model or get_text_embedding_model()

    def summarize(self, text: str, ratio: float = 0.3) -> str:
        """
        Generate extractive summary of text

        Args:
            text: Input text
            ratio: Ratio of sentences to include in summary

        Returns:
            Summary text
        """
        if len(text) <= self.max_length:
            return text

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 3:
            return text[:self.max_length] + "..."

        # Score sentences by position and similarity
        scores = self._score_sentences(sentences)

        # Select top sentences
        num_sentences = max(2, int(len(sentences) * ratio))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        top_indices = sorted(top_indices)  # Keep original order

        # Build summary
        summary = " ".join([sentences[i] for i in top_indices])

        # Truncate if needed
        if len(summary) > self.max_length:
            summary = summary[:self.max_length].rsplit(" ", 1)[0] + "..."

        return summary

    def _score_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences for extractive summarization"""
        # Encode all sentences
        embeddings = self.embedding_model.encode(sentences, show_progress=False)

        # Compute document centroid
        centroid = np.mean(embeddings, axis=0)

        # Score each sentence
        scores = []
        for i, embedding in enumerate(embeddings):
            # Similarity to centroid
            centroid_score = float(np.dot(embedding, centroid))

            # Position score (favor beginning and end)
            position = i / len(sentences)
            position_score = 1.0 - abs(position - 0.3)  # Favor first 30%

            # Length score (prefer medium-length sentences)
            length = len(sentences[i])
            length_score = min(1.0, length / 100) * min(1.0, 300 / max(length, 1))

            # Combined score
            score = 0.5 * centroid_score + 0.3 * position_score + 0.2 * length_score
            scores.append(score)

        return scores


# ============================================================================
# Document Processor
# ============================================================================

class DocumentProcessor:
    """
    Main document processing pipeline
    Combines chunking, summarization, embedding generation, and tagging
    """

    def __init__(self):
        """Initialize document processor with all components"""
        self.chunker = TextChunker()
        self.summarizer = TextSummarizer()
        self.embedding_model = get_text_embedding_model()
        self.tagging_model = get_auto_tagging_model()

    def process_text_document(
        self,
        text: str,
        doc_id: str,
        filename: str = "document.txt"
    ) -> Dict[str, Any]:
        """
        Process a text document through the full pipeline

        Args:
            text: Document text content
            doc_id: Unique document ID
            filename: Original filename

        Returns:
            Dictionary with processed results
        """
        # 1. Generate summary
        summary = self.summarizer.summarize(text)

        # 2. Extract tags
        tags = self.tagging_model.extract_tags(text)

        # 3. Chunk the text
        chunks = self.chunker.chunk_text(text, doc_id)

        # 4. Generate embeddings for chunks
        chunk_texts = [c["content"] for c in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress=False)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "raw_text": text,
            "summary": summary,
            "tags": tags,
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_count": len(chunks),
        }

    def process_image_caption(
        self,
        caption: str,
        detailed_info: Dict[str, str],
        doc_id: str,
        filename: str = "image.jpg"
    ) -> Dict[str, Any]:
        """
        Process an image caption for embedding

        Args:
            caption: Generated image caption
            detailed_info: Detailed caption info from image model
            doc_id: Unique document ID
            filename: Original filename

        Returns:
            Dictionary with processed results
        """
        # Combine caption and detailed description
        combined_text = f"{caption}. {detailed_info.get('detailed_description', '')}"

        # Extract tags from combined text and detected tags
        tags = self.tagging_model.extract_tags(combined_text)
        for tag in detailed_info.get("tags", []):
            tags.append((tag, 0.8))

        # Create a single "chunk" for the image
        chunk = {
            "chunk_id": self.chunker._generate_chunk_id(doc_id, 0),
            "content": combined_text,
            "chunk_number": 0,
            "token_count": len(combined_text) // 4,
        }

        # Generate embedding
        embedding = self.embedding_model.encode_single(combined_text)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "summary": caption,
            "detailed_description": detailed_info.get("detailed_description", ""),
            "tags": tags,
            "chunks": [chunk],
            "embeddings": np.array([embedding]),
            "chunk_count": 1,
        }


# ============================================================================
# File Reader
# ============================================================================

class FileReader:
    """Read various file formats and extract text"""

    @staticmethod
    def read_text_file(file_path: Path) -> str:
        """Read plain text file with encoding detection"""
        import chardet

        with open(file_path, 'rb') as f:
            raw = f.read()
            result = chardet.detect(raw)
            encoding = result.get('encoding', 'utf-8')

        return raw.decode(encoding, errors='ignore')

    @staticmethod
    def read_pdf_file(file_path: Path) -> str:
        """Read PDF file and extract text"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")

    @staticmethod
    def read_docx_file(file_path: Path) -> str:
        """Read DOCX file and extract text"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")

    @classmethod
    def read_file(cls, file_path: Path) -> Tuple[str, str]:
        """
        Read file and extract text based on extension

        Returns:
            Tuple of (text, detected_content_type)
        """
        suffix = file_path.suffix.lower()

        if suffix in ['.txt', '.md']:
            return cls.read_text_file(file_path), 'text'
        elif suffix == '.pdf':
            return cls.read_pdf_file(file_path), 'text'
        elif suffix in ['.doc', '.docx']:
            return cls.read_docx_file(file_path), 'text'
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return '', 'image'
        else:
            # Try as text
            try:
                return cls.read_text_file(file_path), 'text'
            except:
                raise ValueError(f"Unsupported file type: {suffix}")
