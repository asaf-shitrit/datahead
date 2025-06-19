#!/bin/bash

# Music Embedding API Docker Startup Script

echo "🎵 Starting Music Embedding API with Docker Compose..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p ./src/data/vector_db
mkdir -p ./src/data/music_files

# Start the services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service status
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Services are starting up!"
echo ""
echo "🌐 Access your services at:"
echo "   - Music Embedding API: http://localhost:8080"
echo "   - API Documentation: http://localhost:8080/docs"
echo "   - ChromaDB API: http://localhost:8000"
echo "   - ChromaDB Web UI: http://localhost:3000"
echo ""
echo "📝 To view logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose down" 