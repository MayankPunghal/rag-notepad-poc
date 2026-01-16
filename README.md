# RAG Brain

A fully containerized, local multimodal RAG (Retrieval-Augmented Generation) system that runs on CPU only. RAG Brain ingests text and images, performs local text summarization, image captioning, automatic tagging, chunking, and embedding generation. It stores embeddings in FAISS with metadata in SQLite, and exposes a FastAPI backend with semantic search and a chatbot interface powered by Z.AI GLM.

![RAG Brain](https://img.shields.io/badge/version-1.0.0-blue) ![CPU](https://img.shields.io/badge/CPU-only-green) ![Offline](https://img.shields.io/badge/offline-ready-brightgreen)

## Features

- **Multimodal Ingestion**: Upload text files (TXT, MD, PDF, DOCX) and images (JPG, PNG, GIF, BMP, WebP)
- **Local Processing**:
  - Text embeddings using sentence-transformers (all-MiniLM-L6-v2)
  - Image captioning using Florence-2
  - Automatic keyword extraction and tagging
  - Text summarization and chunking
- **Vector Storage**: FAISS for similarity search with SQLite for metadata
- **Semantic Search**: Fast, accurate semantic search across all content
- **RAG Chatbot**: Conversational AI over your documents using GLM API
- **MLflow Tracking**: Log experiments, prompts, and metrics
- **Beautiful Web UI**: Responsive interface for desktop and mobile
- **Fully Local**: Works offline (except optional GLM API calls)

## Architecture

```
rag_brain/
├── backend/
│   ├── core/          # Database, vector store
│   ├── models/        # Embedding models, image captioning
│   ├── services/      # RAG pipeline, chatbot, MLflow
│   └── main.py        # FastAPI application
├── frontend/
│   ├── index.html     # Single-page application
│   └── js/
│       └── app.js     # Frontend logic
├── data/
│   ├── embeddings/    # FAISS indices
│   ├── files/         # Original files
│   └── metadata/      # SQLite database
├── mlflow/            # MLflow experiments
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- (Optional) Z.AI GLM API key for chatbot functionality

### 1. Clone and Setup

```bash
cd rag_brain

# Copy environment template and edit
cp .env.example .env

# Add your GLM API key (get one from https://open.bigmodel.cn/)
# Edit .env and set: GLM_API_KEY=your_api_key_here
```

### 2. Build and Start

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f rag-brain

# Optional: Start MLflow UI
docker-compose --profile mlflow up -d
```

### 3. Access

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI** (optional): http://localhost:5000

## Local Development

### Prerequisites

- Python 3.11+
- 4GB+ RAM

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GLM_API_KEY="your_api_key_here"  # Optional, for chatbot
```

### Running

```bash
# Start the server
python -m backend.main

# Or with uvicorn directly
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

### Web Interface

1. **Upload Documents**: Click "Upload" and drag files or browse
2. **Search**: Use semantic search to find relevant content
3. **Browse**: View all ingested documents
4. **Chat**: Ask questions about your documents

### API Endpoints

#### Health & Stats
- `GET /health` - Health check
- `GET /stats` - System statistics

#### Ingestion
- `POST /api/v1/ingest/text` - Ingest text directly
- `POST /api/v1/ingest/file` - Upload a file

#### Search
- `POST /api/v1/search` - Semantic search
- `GET /api/v1/documents` - List documents
- `GET /api/v1/documents/{id}` - Get document details
- `GET /api/v1/files/{id}` - Get original file
- `DELETE /api/v1/documents/{id}` - Delete document

#### Chat
- `POST /api/v1/chat` - Send chat message
- `POST /api/v1/chat/stream` - Stream chat response
- `GET /api/v1/conversations` - List conversations

### Example: Search via API

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning basics",
    "k": 5,
    "threshold": 0.3
  }'
```

### Example: Chat via API

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "use_rag": true,
    "k": 5
  }'
```

## Configuration

Environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GLM_API_KEY` | Z.AI GLM API key for chatbot | (required for chat) |
| `GLM_API_URL` | GLM API endpoint | https://open.bigmodel.cn/api/paas/v4/chat/completions |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `LOG_LEVEL` | Logging level | INFO |
| `MAX_FILE_SIZE` | Max upload size (bytes) | 52428800 (50MB) |
| `CHUNK_SIZE` | Text chunk size | 512 |
| `CHUNK_OVERLAP` | Chunk overlap | 50 |
| `TOP_K_RESULTS` | Default search results | 5 |

## Getting a GLM API Key

The chatbot uses Z.AI's GLM API for responses:

1. Visit https://open.bigmodel.cn/
2. Register for an account
3. Navigate to API Keys
4. Create a new API key
5. Add it to your `.env` file as `GLM_API_KEY=your_key_here`

**Note**: The RAG system works fully offline for ingestion, search, and retrieval. Only the chatbot responses require the API key.

## MLflow Tracking

Enable MLflow to track experiments:

```bash
# Start with MLflow profile
docker-compose --profile mlflow up -d

# Access MLflow UI at http://localhost:5000
```

Tracked metrics:
- Ingestion: processing time, chunk count, tokens per second
- Retrieval: avg score, retrieval time
- Generation: response time, context length

## Data Persistence

Data is stored in the `./data` directory:

- `data/embeddings/` - FAISS vector indices
- `data/files/` - Original uploaded files
- `data/metadata/` - SQLite database with document/chunk/tag metadata

Backup the entire `data` directory to preserve your knowledge base.

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Reduce `MAX_FILE_SIZE` in settings
2. Use smaller documents
3. Increase Docker memory limit

### Slow First Query

The first search will be slower as models load. Subsequent queries are fast.

### GLM API Errors

Ensure:
1. API key is correctly set in `.env`
2. API key has available credits
3. Network can reach `open.bigmodel.cn`

## Performance Tips

- Use SSD for data directory
- Pre-process large documents before upload
- Batch small documents together
- Enable MLflow for monitoring

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
black .
```

### Building Frontend

The frontend is a single HTML file with inline CSS/JS. No build step required.

## License

MIT License - feel free to use for personal and commercial projects.

## Contributing

Contributions welcome! Please open issues or submit pull requests.

## Acknowledgments

- sentence-transformers for text embeddings
- FAISS by Meta for vector search
- Z.AI for GLM API access
- FastAPI for the web framework
- MLflow for experiment tracking
