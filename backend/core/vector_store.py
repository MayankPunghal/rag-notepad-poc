"""
Vector Store module for RAG Brain
FAISS-based vector store for text and image embeddings
"""

import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import numpy as np

import faiss
from config.settings import settings


# ============================================================================
# FAISS Vector Store
# ============================================================================

class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search
    Supports separate indices for text and image embeddings
    """

    def __init__(
        self,
        index_type: str = None,
        embedding_dim: int = None,
        index_path: Path = None
    ):
        """
        Initialize FAISS vector store

        Args:
            index_type: Type of FAISS index (IndexFlatIP, IndexFlatL2, etc.)
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load index
        """
        self.index_type = index_type or settings.FAISS_INDEX_TYPE
        self.embedding_dim = embedding_dim or settings.TEXT_EMBEDDING_DIM
        self.index_path = index_path or settings.FAISS_TEXT_INDEX_PATH

        # Initialize index
        self.index = self._create_index()

        # Track metadata
        self.next_index = 0  # Next available index
        self.id_to_index: Dict[str, int] = {}  # Maps chunk_id to FAISS index
        self.index_to_id: Dict[int, str] = {}  # Maps FAISS index to chunk_id

        # Try to load existing index
        self._load_if_exists()

    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index"""
        if self.index_type == "IndexFlatIP":
            # Inner product (cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexFlatL2":
            # L2 distance
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IndexIVFFlat":
            # IVF for faster search with large datasets
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, min(100, self.embedding_dim // 4)
            )
        else:
            # Default to IndexFlatIP
            index = faiss.IndexFlatIP(self.embedding_dim)

        return index

    def _load_if_exists(self):
        """Load existing index and metadata from disk"""
        if not self.index_path.exists():
            return

        try:
            # Load the index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            metadata_path = self._get_metadata_path()
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    self.id_to_index = metadata.get("id_to_index", {})
                    self.index_to_id = metadata.get("index_to_id", {})
                    self.next_index = metadata.get("next_index", 0)
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}")

    def _get_metadata_path(self) -> Path:
        """Get path for metadata file"""
        return self.index_path.with_suffix(".metadata")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str]
    ) -> List[int]:
        """
        Add embeddings to the index

        Args:
            embeddings: Array of embeddings shape (n, dim)
            chunk_ids: List of chunk IDs corresponding to embeddings

        Returns:
            List of FAISS indices assigned to the embeddings
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Number of embeddings must match number of chunk IDs")

        # Normalize embeddings for cosine similarity
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings)

        # Assign indices
        indices = []
        for i, chunk_id in enumerate(chunk_ids):
            if chunk_id not in self.id_to_index:
                # New embedding
                idx = self.next_index
                self.id_to_index[chunk_id] = idx
                self.index_to_id[idx] = chunk_id
                indices.append(idx)
                self.next_index += 1
            else:
                # Update existing
                indices.append(self.id_to_index[chunk_id])

        # Add to index
        # For IVF index, need to train first
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            self.index.train(embeddings.astype(np.float32))

        self.index.add(embeddings.astype(np.float32))

        return indices

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = None
    ) -> List[Tuple[int, float, str]]:
        """
        Search for similar embeddings

        Args:
            query_embedding: Query embedding vector shape (1, dim) or (dim,)
            k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, score, chunk_id) tuples
        """
        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize for cosine similarity
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            # Apply threshold
            if threshold and score < threshold:
                continue

            chunk_id = self.index_to_id.get(idx, "")
            results.append((int(idx), float(score), chunk_id))

        return results

    def delete(self, chunk_id: str) -> bool:
        """
        Remove an embedding from the index (note: FAISS doesn't support deletion,
        so we just mark it as removed in metadata)

        Args:
            chunk_id: ID of the chunk to remove

        Returns:
            True if chunk was found and removed
        """
        if chunk_id not in self.id_to_index:
            return False

        idx = self.id_to_index[chunk_id]
        del self.id_to_index[chunk_id]
        if idx in self.index_to_id:
            del self.index_to_id[idx]

        return True

    def save(self):
        """Save index and metadata to disk"""
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, str(self.index_path))

        # Save metadata
        metadata_path = self._get_metadata_path()
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "next_index": self.next_index,
                "index_type": self.index_type,
                "embedding_dim": self.embedding_dim,
                "created_at": datetime.utcnow().isoformat(),
            }, f)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "total_vectors": self.index.ntotal,
            "tracked_ids": len(self.id_to_index),
            "next_index": self.next_index,
            "index_path": str(self.index_path),
        }


class DualVectorStore:
    """
    Manages separate vector stores for text and image embeddings
    """

    def __init__(self):
        """Initialize both text and image vector stores"""
        self.text_store = FAISSVectorStore(
            index_type="IndexFlatIP",
            embedding_dim=settings.TEXT_EMBEDDING_DIM,
            index_path=settings.FAISS_TEXT_INDEX_PATH
        )

        # Image embeddings use same dimension (caption embeddings)
        self.image_store = FAISSVectorStore(
            index_type="IndexFlatIP",
            embedding_dim=settings.TEXT_EMBEDDING_DIM,
            index_path=settings.FAISS_IMAGE_INDEX_PATH
        )

    def add_text_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str]
    ) -> List[int]:
        """Add text embeddings"""
        return self.text_store.add_embeddings(embeddings, chunk_ids)

    def add_image_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str]
    ) -> List[int]:
        """Add image embeddings (from captions)"""
        return self.image_store.add_embeddings(embeddings, chunk_ids)

    def search_text(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = None
    ) -> List[Tuple[int, float, str]]:
        """Search in text embeddings"""
        return self.text_store.search(query_embedding, k, threshold)

    def search_images(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = None
    ) -> List[Tuple[int, float, str]]:
        """Search in image embeddings"""
        return self.image_store.search(query_embedding, k, threshold)

    def search_combined(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = None,
        text_weight: float = 1.0,
        image_weight: float = 0.5
    ) -> List[Tuple[int, float, str, str]]:
        """
        Search both text and image embeddings and combine results

        Returns:
            List of (index, adjusted_score, chunk_id, source) tuples
            where source is 'text' or 'image'
        """
        text_results = self.text_store.search(query_embedding, k, threshold)
        image_results = self.image_store.search(query_embedding, k, threshold)

        # Combine and re-score
        combined = []
        seen_chunks = set()

        for idx, score, chunk_id in text_results:
            if chunk_id not in seen_chunks:
                combined.append((idx, score * text_weight, chunk_id, "text"))
                seen_chunks.add(chunk_id)

        for idx, score, chunk_id in image_results:
            if chunk_id not in seen_chunks:
                combined.append((idx, score * image_weight, chunk_id, "image"))
                seen_chunks.add(chunk_id)
            else:
                # Chunk appears in both, boost its score
                for i, (c_idx, c_score, c_id, c_src) in enumerate(combined):
                    if c_id == chunk_id:
                        combined[i] = (c_idx, c_score + score * image_weight * 0.3, c_id, c_src)
                        break

        # Sort by score
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined[:k]

    def save_all(self):
        """Save both vector stores"""
        self.text_store.save()
        self.image_store.save()

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for both stores"""
        return {
            "text": self.text_store.get_stats(),
            "image": self.image_store.get_stats(),
        }


# ============================================================================
# Global singleton
# ============================================================================

_vector_store: Optional[DualVectorStore] = None


def get_vector_store() -> DualVectorStore:
    """Get the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = DualVectorStore()
    return _vector_store


def init_vector_store():
    """Initialize the vector store"""
    global _vector_store
    _vector_store = DualVectorStore()
    return _vector_store
