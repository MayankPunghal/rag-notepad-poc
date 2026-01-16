"""
Embedding Models module for RAG Brain
Local CPU models for text embeddings and image captioning
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from io import BytesIO
import hashlib

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM

from config.settings import settings


# ============================================================================
# Text Embedding Model
# ============================================================================

class TextEmbeddingModel:
    """
    Local sentence transformer for text embeddings
    CPU-optimized implementation
    """

    def __init__(self, model_name: str = None):
        """
        Initialize text embedding model

        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name or settings.TEXT_EMBEDDING_MODEL
        self.embedding_dim = settings.TEXT_EMBEDDING_DIM
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model"""
        if self._model is None:
            print(f"Loading text embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Array of embeddings shape (n, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding"""
        return self.encode([text])[0]

    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        return float(np.dot(emb1, emb2))


# ============================================================================
# Image Captioning Model
# ============================================================================

class ImageCaptioningModel:
    """
    Local image captioning model for multimodal RAG
    CPU-optimized using Florence-2 or similar lightweight model
    """

    def __init__(self, model_name: str = None):
        """
        Initialize image captioning model

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name or settings.IMAGE_CAPTIONING_MODEL
        self._processor = None
        self._model = None

    @property
    def processor(self):
        """Lazy load processor"""
        if self._processor is None:
            print(f"Loading image captioning processor: {self.model_name}")
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        return self._processor

    @property
    def model(self):
        """Lazy load model"""
        if self._model is None:
            print(f"Loading image captioning model: {self.model_name}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="cpu"  # Force CPU
            )
            # Patch for transformers 4.57+ compatibility with Florence-2
            # The model may be missing _supports_sdpa attribute
            if not hasattr(self._model, '_supports_sdpa'):
                self._model._supports_sdpa = False
            self._model.eval()
        return self._model

    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 100
    ) -> str:
        """
        Generate caption for an image

        Args:
            image: PIL Image
            max_length: Maximum caption length

        Returns:
            Generated caption string
        """
        try:
            # Prepare prompt
            prompt = "<CAPTION>"

            # Process image
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # Generate caption
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_length,
                do_sample=False,
                num_beams=3,
            )

            # Decode caption
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract caption from response (Florence-2 format)
            if ":" in generated_text:
                caption = generated_text.split(":", 1)[1].strip()
            else:
                caption = generated_text

            return caption

        except Exception as e:
            print(f"Error generating caption: {e}")
            # Fallback to simple description
            return f"Image of size {image.size[0]}x{image.size[1]}"

    def generate_detailed_caption(
        self,
        image: Image.Image
    ) -> Dict[str, str]:
        """
        Generate multiple types of captions/descriptions

        Returns:
            Dictionary with caption, detailed_description, and tags
        """
        result = {
            "caption": "",
            "detailed_description": "",
            "tags": []
        }

        try:
            # Generate basic caption
            result["caption"] = self.generate_caption(image)

            # Generate detailed description
            prompt = "<DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=150,
                do_sample=False,
                num_beams=3,
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if ":" in generated_text:
                result["detailed_description"] = generated_text.split(":", 1)[1].strip()
            else:
                result["detailed_description"] = generated_text

            # Generate tags
            prompt = "<OD>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                do_sample=False,
                num_beams=3,
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Extract tags from response
            result["tags"] = self._extract_tags(generated_text)

        except Exception as e:
            print(f"Error generating detailed caption: {e}")
            result["caption"] = "An image"

        return result

    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from model output"""
        # Simple tag extraction - can be enhanced
        tags = []
        if ":" in text:
            tags_part = text.split(":", 1)[1].strip()
            # Split by common delimiters
            for tag in re.split(r'[,;]', tags_part):
                tag = tag.strip().strip("'\"")
                if tag and len(tag) > 2:
                    tags.append(tag)
        return tags[:5]  # Limit to 5 tags


# ============================================================================
# Tagging Model
# ============================================================================

class AutoTaggingModel:
    """
    Automatic keyword extraction and tagging
    Uses rule-based and embedding-based approaches
    """

    def __init__(self, embedding_model: TextEmbeddingModel = None):
        """
        Initialize auto-tagging model

        Args:
            embedding_model: Text embedding model for similarity-based tagging
        """
        self.embedding_model = embedding_model

        # Predefined tag categories
        self.tag_categories = {
            "topic": [
                "technology", "science", "business", "health", "politics",
                "sports", "entertainment", "education", "environment", "travel"
            ],
            "format": [
                "article", "blog", "news", "tutorial", "documentation",
                "reference", "opinion", "analysis", "summary", "review"
            ],
            "content": [
                "guide", "how-to", "explanation", "comparison", "list",
                "faq", "introduction", "overview", "example", "template"
            ]
        }

        # Common keyword patterns
        self.keyword_patterns = {
            r"\b(Python|JavaScript|Java|C\+\+|Go|Rust|SQL)\b": "programming-language",
            r"\b(API|REST|GraphQL|HTTP|JSON|XML)\b": "web-technology",
            r"\b(machine learning|AI|deep learning|neural network)\b": "artificial-intelligence",
            r"\b(database|SQL|NoSQL|MongoDB|PostgreSQL)\b": "database",
            r"\b(cloud|AWS|Azure|GCP|Docker|Kubernetes)\b": "cloud-computing",
            r"\b(security|encryption|authentication|authorization)\b": "security",
        }

    def extract_tags(
        self,
        text: str,
        max_tags: int = 5,
        min_score: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Extract tags from text

        Args:
            text: Input text
            max_tags: Maximum number of tags to return
            min_score: Minimum confidence score

        Returns:
            List of (tag, score) tuples
        """
        tags = []

        # 1. Pattern-based keyword extraction
        for pattern, category in self.keyword_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in set(matches):
                    tags.append((match.lower(), 0.9))

        # 2. Category similarity tagging
        if self.embedding_model:
            text_embedding = self.embedding_model.encode_single(text[:500])

            for category, tag_list in self.tag_categories.items():
                best_tag = None
                best_score = 0

                for tag in tag_list:
                    tag_embedding = self.embedding_model.encode_single(tag)
                    score = float(np.dot(text_embedding, tag_embedding))

                    if score > best_score and score > min_score:
                        best_score = score
                        best_tag = tag

                if best_tag:
                    tags.append((best_tag, best_score))

        # 3. Named entity extraction (simple)
        # Extract capitalized words that appear multiple times
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]:
            if freq >= 2:
                tags.append((word.lower(), 0.7))

        # Sort by score and limit
        tags = sorted(tags, key=lambda x: x[1], reverse=True)[:max_tags]

        # Deduplicate
        seen = set()
        unique_tags = []
        for tag, score in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append((tag, score))

        return unique_tags


# ============================================================================
# Image Utilities
# ============================================================================

class ImageUtils:
    """Utility functions for image processing"""

    @staticmethod
    def load_image(image_bytes: bytes) -> Image.Image:
        """Load image from bytes"""
        return Image.open(BytesIO(image_bytes))

    @staticmethod
    def get_image_hash(image_bytes: bytes) -> str:
        """Generate hash for image deduplication"""
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    @staticmethod
    def resize_image(
        image: Image.Image,
        max_size: Tuple[int, int] = (1024, 1024),
        keep_aspect_ratio: bool = True
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        if keep_aspect_ratio:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        else:
            image = image.resize(max_size, Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def get_image_info(image: Image.Image) -> Dict[str, Any]:
        """Get image information"""
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": len(image.tobytes()) if hasattr(image, 'tobytes') else 0,
        }


# ============================================================================
# Global model instances
# ============================================================================

_text_embedding_model: Optional[TextEmbeddingModel] = None
_image_captioning_model: Optional[ImageCaptioningModel] = None
_auto_tagging_model: Optional[AutoTaggingModel] = None


def get_text_embedding_model() -> TextEmbeddingModel:
    """Get global text embedding model instance"""
    global _text_embedding_model
    if _text_embedding_model is None:
        _text_embedding_model = TextEmbeddingModel()
    return _text_embedding_model


def get_image_captioning_model() -> ImageCaptioningModel:
    """Get global image captioning model instance"""
    global _image_captioning_model
    if _image_captioning_model is None:
        _image_captioning_model = ImageCaptioningModel()
    return _image_captioning_model


def get_auto_tagging_model() -> AutoTaggingModel:
    """Get global auto-tagging model instance"""
    global _auto_tagging_model
    if _auto_tagging_model is None:
        _auto_tagging_model = AutoTaggingModel(get_text_embedding_model())
    return _auto_tagging_model


def init_all_models():
    """Initialize all models (call at startup for faster subsequent access)"""
    get_text_embedding_model().model  # Load text model
    # Image model is lazy-loaded
    return {
        "text_model": get_text_embedding_model().model_name,
        "image_model": get_image_captioning_model().model_name,
    }
