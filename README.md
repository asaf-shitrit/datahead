# Music Embedding API

A FastAPI wrapper for a music embedding pipeline that can upload music files and find similar music using vector similarity search. **Now with MCP (Model Context Protocol) support for LLM integration!**

## Features

- 🎵 **Music File Upload**: Upload music files (MP3, WAV, FLAC, M4A, etc.) and automatically generate embeddings
- 🔍 **Similarity Search**: Find similar music using uploaded files or existing file IDs
- 🐳 **Docker Support**: Complete containerized setup with ChromaDB
- 📊 **Vector Database**: Uses ChromaDB for efficient similarity search
- 🎯 **Audio Processing**: Advanced audio segmentation and feature extraction
- 🔧 **RESTful API**: Clean REST API with automatic documentation
- 🤖 **MCP Server**: Model Context Protocol server for LLM integration

## Project Structure

```
datahead/
├── src/                    # Source code
│   ├── api/               # API components
│   │   ├── api.py         # FastAPI application
│   │   ├── mcp_server.py  # MCP server
│   │   └── mcp_config.json
│   ├── core/              # Core pipeline components
│   │   ├── audio_processor.py
│   │   ├── embedding_generator.py
│   │   ├── ingestion_pipeline.py
│   │   └── vector_store.py
│   ├── utils/             # Utilities and configuration
│   │   ├── config.py
│   │   └── start_api.py
│   ├── main.py            # Main entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   └── setup.py
├── scripts/               # Utility scripts
│   ├── main.py           # Pipeline CLI
│   └── example_usage.py
├── tests/                 # Test files
│   ├── test_pipeline.py
│   └── test_mcp.py
├── docs/                  # Documentation
│   └── README.md
├── docker-compose.yml     # Docker setup
├── start_docker.sh        # Docker startup script
└── README.md             # This file
```

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB of available RAM

### 1. Clone and Setup

```bash
git clone <your-repo>
cd datahead
chmod +x start_docker.sh
```

### 2. Start Services

```bash
./start_docker.sh
```

This will start:
- **ChromaDB** on port 8000
- **Music Embedding API** on port 8080
- **ChromaDB Web UI** on port 3000

### 3. Access the API

- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **ChromaDB Web UI**: http://localhost:3000

## Local Development

### Prerequisites

- Python 3.11+
- FFmpeg installed
- ChromaDB (can be run via Docker)

### 1. Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

### 2. Start ChromaDB (Docker)

```bash
docker-compose up chromadb -d
```

### 3. Run the Application

#### Option A: Using the main entry point
```bash
# Run FastAPI server
python src/main.py --mode api

# Run MCP server
python src/main.py --mode mcp

# Run pipeline directly
python src/main.py --mode pipeline
```

#### Option B: Run individual components
```bash
# FastAPI server
python src/utils/start_api.py

# MCP server
python src/api/mcp_server.py

# Pipeline CLI
python scripts/main.py
```

## MCP (Model Context Protocol) Server

The Music Embedding API also includes an MCP server that allows LLMs to use music similarity search as tools. This enables natural language interactions with your music database.

### MCP Tools Available

1. **`upload_music_file`** - Upload and process music files
2. **`search_similar_music`** - Find similar music using a query file
3. **`search_by_file_id`** - Search similar music using a database file ID
4. **`get_file_info`** - Get detailed information about a file
5. **`list_all_files`** - List all files in the database
6. **`delete_file`** - Delete a file from the database

### Using the MCP Server

#### 1. Start the MCP Server

```bash
python src/main.py --mode mcp
```

#### 2. Configure MCP Client

Use the provided `src/api/mcp_config.json` to configure your MCP client:

```json
{
  "mcpServers": {
    "music-embedding": {
      "command": "python",
      "args": ["src/api/mcp_server.py"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

#### 3. LLM Integration Examples

**Example 1: Upload and Search**
```
User: "I have a song called song.mp3, can you find similar music?"
LLM: I'll help you find similar music! Let me first upload your song and then search for similar pieces.
     [Calls upload_music_file with song.mp3]
     [Calls search_similar_music with the uploaded file]
```

**Example 2: Get File Information**
```
User: "What's the tempo of the song with ID song_segment_0?"
LLM: Let me get the detailed information about that song for you.
     [Calls get_file_info with song_segment_0]
```

**Example 3: Database Overview**
```
User: "Show me all the music in the database"
LLM: I'll show you an overview of all the music files in the database.
     [Calls list_all_files]
```

### Test the MCP Server

```bash
python tests/test_mcp.py
```

This will show you all available tools and example usage scenarios.

## API Endpoints

### Upload Music File
```http
POST /upload
Content-Type: multipart/form-data

file: [music file]
```

**Response:**
```json
{
  "success": true,
  "file_id": "song_segment_0",
  "segments_created": 3,
  "message": "Successfully processed song.mp3",
  "metadata": {
    "filename": "song.mp3",
    "file_size": 5242880,
    "duration": 180.5,
    "segments": [
      {
        "segment_id": "song_segment_0",
        "duration": 30.0,
        "tempo": 120.5
      }
    ]
  }
}
```

### Search Similar Music
```http
POST /search?n_results=10
Content-Type: multipart/form-data

file: [query music file]
```

**Response:**
```json
{
  "success": true,
  "query_file_id": "query.mp3",
  "results": [
    {
      "id": "uuid-123",
      "distance": 0.234,
      "metadata": {
        "filename": "similar_song.mp3",
        "file_path": "/path/to/similar_song.mp3",
        "segment_index": 0,
        "segment_duration": 30.0,
        "tempo": 118.2,
        "num_beats": 60,
        "file_duration": 180.5
      }
    }
  ],
  "total_results": 10,
  "message": "Found 10 similar music pieces"
}
```

### Search by File ID
```http
GET /search/{file_id}?n_results=10
```

### Get File Details
```http
GET /files/{file_id}
```

### List All Files
```http
GET /files
```

### Delete File
```http
DELETE /files/{file_id}
```

## Configuration

The application can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port |
| `VECTOR_DB_PATH` | `./vector_db` | Local vector database path |
| `MUSIC_FILES_PATH` | `./music_files` | Music files directory |

## Docker Commands

### Start Services
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f music-api
docker-compose logs -f chromadb
```

### Stop Services
```bash
docker-compose down
```

### Rebuild and Start
```bash
docker-compose up --build -d
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   ChromaDB      │    │   Audio Files   │
│   (Port 8080)   │◄──►│   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Upload        │    │   Vector        │    │   Audio         │
│   Endpoint      │    │   Storage       │    │   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Search        │    │   Embedding     │    │   Feature       │
│   Endpoint      │    │   Generation    │    │   Extraction    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Server    │    │   LLM Tools     │    │   Natural       │
│   (stdio)       │    │   Interface     │    │   Language      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- FLAC (.flac)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)
- WMA (.wma)

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8080
   # Kill the process or change the port in docker-compose.yml
   ```

2. **ChromaDB Connection Issues**
   ```bash
   # Check if ChromaDB is running
   docker-compose ps
   # Restart ChromaDB
   docker-compose restart chromadb
   ```

3. **Audio Processing Errors**
   - Ensure FFmpeg is installed in the Docker container
   - Check audio file format support
   - Verify file is not corrupted

4. **MCP Server Issues**
   - Ensure MCP dependency is installed: `pip install mcp`
   - Check that the server is running: `python src/main.py --mode mcp`
   - Verify MCP client configuration

5. **Import Errors**
   - Make sure you're running from the project root
   - Check that all `__init__.py` files are present
   - Verify Python path includes the `src` directory

### Logs

View detailed logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f music-api
docker-compose logs -f chromadb
```

## Development

### Adding New Endpoints

1. Edit `src/api/api.py`
2. Add new route handlers
3. Update Pydantic models if needed
4. Test with the interactive docs at `/docs`

### Adding New MCP Tools

1. Edit `src/api/mcp_server.py`
2. Add new tool definition in `list_tools()`
3. Add corresponding handler method
4. Update test script `tests/test_mcp.py`

### Modifying Audio Processing

1. Edit `src/core/audio_processor.py`
2. Update feature extraction in `src/core/embedding_generator.py`
3. Test with sample audio files

### Database Schema Changes

1. Update metadata structure in `src/core/ingestion_pipeline.py`
2. Consider migration strategy for existing data
3. Test with sample data

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request 