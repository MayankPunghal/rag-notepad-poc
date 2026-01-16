"""
GLM Chatbot module for RAG Brain
Integrates with Z.AI GLM API for chatbot responses
Includes mock mode for local testing without API
"""

import os
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime
import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from backend.services.rag_pipeline import get_rag_pipeline
from backend.services.mlflow_tracker import get_mlflow_tracker


# ============================================================================
# Mock Response Generator
# ============================================================================

class MockChatGenerator:
    """Generates mock responses for testing without GLM API"""

    @staticmethod
    def generate_response(query: str, rag_context: str = "") -> str:
        """Generate a contextual mock response based on query and RAG context"""

        # Check if we have RAG context to use
        if rag_context and len(rag_context.strip()) > 100:
            return f"""Based on the documents I found, here's what I can tell you about "{query}":

{MockChatGenerator._extract_relevant_info(rag_context, query)}

This information comes from the knowledge base. Is there anything specific you'd like me to elaborate on?"""

        # Generic responses without RAG context
        query_lower = query.lower()

        if any(w in query_lower for w in ["hello", "hi", "hey"]):
            return "Hello! I'm your RAG Brain assistant. I can help you search and answer questions using your document knowledge base. Try asking me something about your uploaded documents!"

        elif any(w in query_lower for w in ["who are you", "what are you"]):
            return "I'm RAG Brain, an AI assistant that helps you search and retrieve information from your uploaded documents. I use semantic search and retrieval-augmented generation to provide accurate answers based on your knowledge base."

        elif any(w in query_lower for w in ["help", "how do i"]):
            return """I can help you with:
• **Search** - Ask questions and I'll find relevant documents
• **Summarize** - Get summaries of your documents
• **Explore** - Discover connections between your documents
• **Answer** - Get specific answers from your knowledge base

Try uploading some documents first, then ask me questions about them!"""

        elif any(w in query_lower for w in ["thank", "thanks"]):
            return "You're welcome! Feel free to ask if you have more questions."

        else:
            return f"""I understand you're asking about "{query}".

To give you the most accurate answer, I'd need to search through your documents. Here are some tips:

1. **Upload documents** first if you haven't already
2. **Be specific** in your questions
3. **Use keywords** from your documents

Currently, I'm running in **mock mode** without the GLM API connected. For full AI-powered responses, please add credits to your GLM account or configure a valid API key.

Is there anything else I can help you with?"""

    @staticmethod
    def _extract_relevant_info(context: str, query: str) -> str:
        """Extract relevant snippets from RAG context"""
        # Simple extraction - in real mode this would be done by the LLM
        lines = context.split('\n')
        relevant_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]
        return '\n\n'.join(relevant_lines[:3]) if relevant_lines else "Relevant information found in your documents."


# ============================================================================
# GLM API Client
# ============================================================================

class GLMClient:
    """
    Client for Z.AI GLM API
    Supports chat completion with streaming
    """

    def __init__(
        self,
        api_key: str = None,
        api_url: str = None,
        model: str = "glm-4"
    ):
        """
        Initialize GLM client

        Args:
            api_key: GLM API key
            api_url: GLM API endpoint URL
            model: Model name (glm-4, glm-3-turbo, etc.)
        """
        self.api_key = api_key or settings.GLM_API_KEY
        self.api_url = api_url or settings.GLM_API_URL
        self.model = model
        self.client = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy initialize HTTP client"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self.client

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Build API request payload"""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or settings.RAG_TEMPERATURE,
            "max_tokens": max_tokens or settings.RAG_MAX_TOKENS,
            "stream": stream,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException))
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Get chat completion from GLM API

        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            API response dictionary
        """
        if not self.api_key:
            raise ValueError("GLM API key is not configured. Please set GLM_API_KEY environment variable.")

        payload = self._build_payload(messages, temperature, max_tokens, stream=False)

        response = await self.http_client.post(
            self.api_url,
            headers=self._get_headers(),
            json=payload,
        )
        response.raise_for_status()

        return response.json()

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> AsyncIterator[str]:
        """
        Get streaming chat completion from GLM API

        Yields:
            Response text chunks
        """
        if not self.api_key:
            raise ValueError("GLM API key is not configured. Please set GLM_API_KEY environment variable.")

        payload = self._build_payload(messages, temperature, max_tokens, stream=True)

        async with self.http_client.stream(
            "POST",
            self.api_url,
            headers=self._get_headers(),
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None


# ============================================================================
# Anthropic Claude API Client
# ============================================================================

class ClaudeClient:
    """
    Client for Anthropic Claude API
    Supports chat completion with streaming
    """

    def __init__(
        self,
        api_key: str = None,
        api_url: str = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key
            api_url: Custom API URL (e.g., Z.AI proxy)
            model: Model name (claude-3-5-sonnet-20241022, claude-3-haiku-20240307, etc.)
        """
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.api_url = api_url or settings.ANTHROPIC_API_URL
        self.model = model
        self.client = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy initialize HTTP client"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self.client

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _convert_to_claude_format(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple:
        """
        Convert OpenAI-style messages to Claude format

        Returns:
            Tuple of (system_message, claude_messages)
        """
        system_message = ""
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return system_message, claude_messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException))
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        Get chat completion from Claude API

        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            API response dictionary
        """
        if not self.api_key:
            raise ValueError("Anthropic API key is not configured. Please set ANTHROPIC_API_KEY environment variable.")

        system_message, claude_messages = self._convert_to_claude_format(messages)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens or settings.RAG_MAX_TOKENS,
            "temperature": temperature or settings.RAG_TEMPERATURE,
            "messages": claude_messages,
        }

        if system_message:
            payload["system"] = system_message

        response = await self.http_client.post(
            self.api_url,
            headers=self._get_headers(),
            json=payload,
        )
        response.raise_for_status()

        data = response.json()

        # Convert Claude response to OpenAI-style format for compatibility
        content_array = data.get("content", [])
        if content_array and len(content_array) > 0:
            content_text = content_array[0].get("text", "")
        else:
            content_text = ""

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content_text
                }
            }],
            "usage": data.get("usage", {}),
            "model": data.get("model", self.model),
        }

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> AsyncIterator[str]:
        """
        Get streaming chat completion from Claude API

        Yields:
            Response text chunks
        """
        if not self.api_key:
            raise ValueError("Anthropic API key is not configured. Please set ANTHROPIC_API_KEY environment variable.")

        system_message, claude_messages = self._convert_to_claude_format(messages)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens or settings.RAG_MAX_TOKENS,
            "temperature": temperature or settings.RAG_TEMPERATURE,
            "messages": claude_messages,
            "stream": True,
        }

        if system_message:
            payload["system"] = system_message

        async with self.http_client.stream(
            "POST",
            self.api_url,
            headers=self._get_headers(),
            json=payload,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    if chunk.get("type") == "content_block_delta":
                        delta = chunk.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
                except (json.JSONDecodeError, KeyError):
                    continue

    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None


# ============================================================================
# RAG Chatbot
# ============================================================================

class RAGChatbot:
    """
    RAG-enabled chatbot supporting multiple LLM providers
    Combines retrieval with generation
    """

    def __init__(self, model: str = None, provider: str = None):
        """
        Initialize RAG chatbot

        Args:
            model: Model name (auto-detected based on provider if not specified)
            provider: LLM provider ('anthropic' or 'glm', defaults to CHAT_PROVIDER setting)
        """
        self.provider = provider or settings.CHAT_PROVIDER
        self.rag_pipeline = get_rag_pipeline()
        self.mlflow_tracker = get_mlflow_tracker()

        # Initialize the appropriate client based on provider
        if self.provider == "anthropic":
            self.model = model or "claude-3-5-sonnet-20241022"
            self.client = ClaudeClient(model=self.model)
        else:  # glm
            self.model = model or "glm-4"
            self.client = GLMClient(model=self.model)

    async def _prepare_chat_messages(
        self,
        query: str,
        use_rag: bool = True,
        k: int = None
    ) -> tuple:
        """
        Prepare messages for GLM API call

        Returns:
            Tuple of (messages, context_docs, rag_context)
        """
        context_docs = []
        rag_context = ""

        if use_rag:
            # Get ALL retrieved chunks (no deduplication) for complete context
            k = k if k and k > 0 else 10  # Use provided k or default to 10

            # Generate RAG context using ALL chunks
            rag_context = await self.rag_pipeline.generate_rag_context(query, k=k)

            # Get retrieved chunks for tracking (use all chunks, not just best per doc)
            retrieved_chunks = await self.rag_pipeline.retrieve_all_chunks(query, k=k)
            # Count unique docs for tracking
            context_docs = list(set(r["doc_id"] for r in retrieved_chunks))

            # Debug logging
            print(f"[RAG DEBUG] Query: {query[:50]}...")
            print(f"[RAG DEBUG] Retrieved {len(retrieved_chunks)} chunks from {len(context_docs)} documents")
            print(f"[RAG DEBUG] Context length: {len(rag_context)} chars")
            print(f"[RAG DEBUG] Context preview: {rag_context[:200]}...")

        # Build messages
        messages = []

        # System message
        if use_rag:
            messages.append({
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on "
                    "the provided context from a knowledge base. Use the context to "
                    "inform your answers, but respond naturally as if you know this "
                    "information. If the context doesn't contain relevant information, "
                    "say so honestly."
                )
            })
            messages.append({
                "role": "user",
                "content": f"Context:\n{rag_context}\n\nQuestion: {query}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
            messages.append({
                "role": "user",
                "content": query
            })

        return messages, context_docs, rag_context

    async def chat(
        self,
        query: str,
        conversation_id: str = None,
        use_rag: bool = True,
        k: int = None
    ) -> Dict[str, Any]:
        """
        Process a chat query with RAG (non-streaming)

        Args:
            query: User query
            conversation_id: Optional conversation ID for context
            use_rag: Whether to use RAG retrieval
            k: Number of documents to retrieve

        Returns:
            Response dictionary with answer and metadata
        """
        start_time = datetime.utcnow()
        context_docs = []
        rag_context = ""

        try:
            # Prepare messages
            messages, context_docs, rag_context = await self._prepare_chat_messages(
                query, use_rag, k
            )

            generation_start = datetime.utcnow()

            # Get response from GLM
            api_response = await self.client.chat_completion(messages)
            response_text = api_response.get("choices", [{}])[0].get("message", {}).get("content", "")

            generation_time = (datetime.utcnow() - generation_start).total_seconds()
            total_time = (datetime.utcnow() - start_time).total_seconds()

            # Track in MLflow
            self.mlflow_tracker.track_generation(
                query=query,
                context=rag_context,
                response=response_text,
                context_docs=context_docs,
                generation_time=generation_time,
                model=self.model
            )

            # Save to database if conversation_id provided
            if conversation_id:
                await self._save_conversation_message(
                    conversation_id, query, response_text, context_docs
                )

            return {
                "query": query,
                "response": response_text,
                "context_docs": context_docs,
                "context_used": use_rag,
                "generation_time": generation_time,
                "total_time": total_time,
                "model": self.model,
            }

        except Exception as e:
            # Fall back to mock mode when GLM API fails
            total_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)

            # Use mock generator as fallback
            generation_start = datetime.utcnow()
            response_text = MockChatGenerator.generate_response(query, rag_context)
            generation_time = (datetime.utcnow() - generation_start).total_seconds()
            total_time = (datetime.utcnow() - start_time).total_seconds()

            # Track mock generation in MLflow
            self.mlflow_tracker.track_generation(
                query=query,
                context=rag_context,
                response=response_text,
                context_docs=context_docs,
                generation_time=generation_time,
                model=f"{self.model}-mock"
            )

            # Save to database if conversation_id provided
            if conversation_id:
                await self._save_conversation_message(
                    conversation_id, query, response_text, context_docs
                )

            return {
                "query": query,
                "response": response_text,
                "context_docs": context_docs,
                "context_used": use_rag,
                "generation_time": generation_time,
                "total_time": total_time,
                "model": f"{self.model}-mock",
                "mock_mode": True,
                "original_error": error_msg,
            }

    async def chat_stream(
        self,
        query: str,
        conversation_id: str = None,
        use_rag: bool = True,
        k: int = None
    ):
        """
        Process a chat query with RAG (streaming)

        Yields:
            Dictionaries with streaming response chunks
        """
        start_time = datetime.utcnow()
        context_docs = []

        try:
            # Prepare messages
            messages, context_docs, rag_context = await self._prepare_chat_messages(
                query, use_rag, k
            )

            generation_start = datetime.utcnow()
            response_text = ""

            # Stream response from GLM
            async for chunk in self.client.chat_completion_stream(messages):
                response_text += chunk
                yield {
                    "content": chunk,
                    "done": False,
                    "context_docs": context_docs,
                }

            generation_time = (datetime.utcnow() - generation_start).total_seconds()
            total_time = (datetime.utcnow() - start_time).total_seconds()

            # Track in MLflow
            self.mlflow_tracker.track_generation(
                query=query,
                context=rag_context,
                response=response_text,
                context_docs=context_docs,
                generation_time=generation_time,
                model=self.model
            )

            # Save to database if conversation_id provided
            if conversation_id:
                await self._save_conversation_message(
                    conversation_id, query, response_text, context_docs
                )

            # Final yield with completion
            yield {
                "content": "",
                "done": True,
                "context_docs": context_docs,
                "full_response": response_text,
            }

        except Exception as e:
            # Fall back to mock mode for streaming too
            error_msg = str(e)
            generation_start = datetime.utcnow()
            response_text = MockChatGenerator.generate_response(query, rag_context)
            generation_time = (datetime.utcnow() - generation_start).total_seconds()

            # Track mock generation
            self.mlflow_tracker.track_generation(
                query=query,
                context=rag_context,
                response=response_text,
                context_docs=context_docs,
                generation_time=generation_time,
                model=f"{self.model}-mock"
            )

            # Save to database if conversation_id provided
            if conversation_id:
                await self._save_conversation_message(
                    conversation_id, query, response_text, context_docs
                )

            # Yield mock response
            yield {
                "content": response_text,
                "done": True,
                "context_docs": context_docs,
                "full_response": response_text,
                "mock_mode": True,
                "original_error": error_msg,
            }

    async def _save_conversation_message(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        context_docs: List[str]
    ):
        """Save conversation messages to database"""
        from backend.core.database import ConversationRepository, get_db_session

        async with get_db_session() as session:
            # Get or create conversation
            conv = await ConversationRepository.get_by_id(session, conversation_id)
            if not conv:
                conv = await ConversationRepository.create(session)

            # Add user message
            await ConversationRepository.add_message(
                session, conversation_id, "user", user_message
            )

            # Add assistant message
            await ConversationRepository.add_message(
                session, conversation_id, "assistant", assistant_message, context_docs
            )

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation details"""
        from backend.core.database import ConversationRepository, get_db_session

        async with get_db_session() as session:
            conv = await ConversationRepository.get_by_id(session, conversation_id)
            if not conv:
                return None

            messages = await ConversationRepository.get_messages(session, conversation_id)

            return {
                **conv.to_dict(),
                "messages": [m.to_dict() for m in messages],
            }

    async def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all conversations"""
        from backend.core.database import ConversationRepository, get_db_session

        async with get_db_session() as session:
            convs = await ConversationRepository.list_all(session, limit)
            return [c.to_dict() for c in convs]

    async def create_conversation(self, title: str = None) -> Dict[str, Any]:
        """Create a new conversation"""
        from backend.core.database import ConversationRepository, get_db_session

        title = title or f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"

        async with get_db_session() as session:
            conv = await ConversationRepository.create(session, title)
            return conv.to_dict()

    async def generate_image_description(self, basic_caption: str, filename: str = "image") -> Dict[str, Any]:
        """
        Generate detailed AI description and tags for an image using GLM API

        Args:
            basic_caption: Basic caption from Florence-2 model
            filename: Original filename (used as fallback when caption is poor)

        Returns:
            Dictionary with detailed_description, tags, and summary
        """
        # If caption is empty or too generic, use filename for context
        if not basic_caption or basic_caption.lower() in ["an image", "image", "a photo", "photo", ""]:
            basic_caption = f"A file named '{filename}'"

        prompt = f"""Analyze this image and generate a detailed description and relevant tags.

Image information: {basic_caption}

Please provide your response in the following JSON format:
{{
    "detailed_description": "A comprehensive 2-3 sentence description of what is shown in the image, including colors, objects, setting, mood, and any notable details",
    "summary": "A brief one-sentence summary of the image",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

Generate 5-8 relevant tags that describe:
- Main objects or subjects
- Colors and visual elements
- Style or mood
- Context or setting
- Any actions or activities

Respond ONLY with valid JSON, no other text."""

        try:
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON response
            import re
            json_match = re.search(r'\{[^{}]*\{.*\}[^{}]*\}|\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {
                    "detailed_description": result.get("detailed_description", basic_caption),
                    "summary": result.get("summary", basic_caption),
                    "tags": result.get("tags", [])
                }
            else:
                # Fallback: extract info from text response
                return {
                    "detailed_description": response_text,
                    "summary": basic_caption,
                    "tags": self._extract_tags_from_text(response_text)
                }

        except Exception as e:
            print(f"Error generating image description: {e}")
            # Fallback to basic caption with simple tag extraction
            return {
                "detailed_description": f"An image showing: {basic_caption}",
                "summary": basic_caption,
                "tags": self._extract_tags_from_text(basic_caption)
            }

    def _extract_tags_from_text(self, text: str) -> List[str]:
        """Extract potential tags from text"""
        import re
        # Simple keyword extraction
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', text)
        # Common non-descriptive words to filter out
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'them', 'their', 'about', 'which', 'when', 'what', 'where', 'will', 'your', 'just', 'like', 'more', 'some', 'such', 'into', 'over', 'also'}
        tags = [w.capitalize() for w in set(words) if len(w) > 3 and w.lower() not in stop_words][:8]
        return tags

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        from backend.core.database import ConversationRepository, get_db_session

        async with get_db_session() as session:
            conv = await ConversationRepository.get_by_id(session, conversation_id)
            if conv:
                await session.delete(conv)
                return True
            return False


# ============================================================================
# Global chatbot instance
# ============================================================================

_glm_chatbot: Optional[RAGChatbot] = None


def get_chatbot() -> RAGChatbot:
    """Get the global chatbot instance"""
    global _glm_chatbot
    if _glm_chatbot is None:
        _glm_chatbot = RAGChatbot()
    return _glm_chatbot


def init_chatbot():
    """Initialize the chatbot"""
    global _glm_chatbot
    _glm_chatbot = RAGChatbot()
    return _glm_chatbot
