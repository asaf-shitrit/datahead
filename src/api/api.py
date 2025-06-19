#!/usr/bin/env python3
"""
FastAPI wrapper for Music Embedding Pipeline
Provides REST API endpoints for file upload and similarity search
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from ..core.ingestion_pipeline import MusicIngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Music Embedding API",
    description="API for uploading music files and finding similar music",
    version="1.0.0"
)

# Initialize pipeline
pipeline = MusicIngestionPipeline()

# Pydantic models for request/response
class UploadResponse(BaseModel):
    success: bool
    file_id: Optional[str] = None
    segments_created: int = 0
    message: str
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    success: bool
    query_file_id: Optional[str] = None
    results: List[Dict[str, Any]] = []
    total_results: int = 0
    message: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    message: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Music Embedding API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if vector store is accessible
        stats = pipeline.vector_store.get_collection_stats()
        return {
            "status": "healthy",
            "vector_store": {
                "total_embeddings": stats.get("total_embeddings", 0),
                "embedding_dimension": stats.get("embedding_dimension", 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a music file and process it into embeddings
    
    Args:
        file: Music file to upload (supports: mp3, wav, flac, m4a, etc.)
        
    Returns:
        Upload response with processing results
    """
    try:
        # Validate file type
        supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary location
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process the file
            logger.info(f"Processing uploaded file: {file.filename}")
            result = pipeline.process_single_file(temp_file_path)
            
            if not result:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to process the uploaded file"
                )
            
            # Save to vector store
            save_success = pipeline.save_to_vector_store([result])
            
            if not save_success:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to save embeddings to vector store"
                )
            
            # Generate a unique file ID (using the first segment ID)
            file_id = result['segments'][0]['segment_id'] if result['segments'] else None
            
            return UploadResponse(
                success=True,
                file_id=file_id,
                segments_created=len(result['segments']),
                message=f"Successfully processed {file.filename}",
                metadata={
                    'filename': file.filename,
                    'file_size': result['metadata'].get('file_size', 0),
                    'duration': result['metadata'].get('duration', 0),
                    'segments': [
                        {
                            'segment_id': seg['segment_id'],
                            'duration': seg['segment_metadata']['segment_duration'],
                            'tempo': seg['segment_metadata']['tempo']
                        }
                        for seg in result['segments']
                    ]
                }
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_similar(
    file: UploadFile = File(...),
    n_results: int = Query(default=10, ge=1, le=50, description="Number of similar results to return")
):
    """
    Search for similar music using an uploaded query file
    
    Args:
        file: Query music file to find similar music for
        n_results: Number of similar results to return (1-50)
        
    Returns:
        Search response with similar music results
    """
    try:
        # Validate file type
        supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary location
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Search for similar music
            logger.info(f"Searching for music similar to: {file.filename}")
            similar_results = pipeline.search_similar_music(temp_file_path, n_results=n_results)
            
            if not similar_results:
                return SearchResponse(
                    success=True,
                    query_file_id=None,
                    results=[],
                    total_results=0,
                    message="No similar music found"
                )
            
            # Format results for response
            formatted_results = []
            for result in similar_results:
                formatted_result = {
                    'id': result['id'],
                    'distance': result['distance'],
                    'metadata': {
                        'filename': result['metadata']['filename'],
                        'file_path': result['metadata']['file_path'],
                        'segment_index': result['metadata']['segment_index'],
                        'segment_duration': result['metadata']['segment_duration'],
                        'tempo': result['metadata']['tempo'],
                        'num_beats': result['metadata']['num_beats'],
                        'file_duration': result['metadata']['file_duration']
                    }
                }
                formatted_results.append(formatted_result)
            
            return SearchResponse(
                success=True,
                query_file_id=file.filename,
                results=formatted_results,
                total_results=len(formatted_results),
                message=f"Found {len(formatted_results)} similar music pieces"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching similar music: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/{file_id}", response_model=SearchResponse)
async def search_by_id(
    file_id: str,
    n_results: int = Query(default=10, ge=1, le=50, description="Number of similar results to return")
):
    """
    Search for similar music using a file ID from the database
    
    Args:
        file_id: ID of the file in the database to search for similar music
        n_results: Number of similar results to return (1-50)
        
    Returns:
        Search response with similar music results
    """
    try:
        # Get embedding by ID
        embedding_data = pipeline.vector_store.get_embedding_by_id(file_id)
        
        if not embedding_data:
            raise HTTPException(
                status_code=404, 
                detail=f"File with ID {file_id} not found in database"
            )
        
        # Search for similar embeddings
        similar_results = pipeline.vector_store.search_similar(
            embedding_data['embedding'], n_results=n_results
        )
        
        if not similar_results:
            return SearchResponse(
                success=True,
                query_file_id=file_id,
                results=[],
                total_results=0,
                message="No similar music found"
            )
        
        # Format results for response
        formatted_results = []
        for result in similar_results:
            formatted_result = {
                'id': result['id'],
                'distance': result['distance'],
                'metadata': {
                    'filename': result['metadata']['filename'],
                    'file_path': result['metadata']['file_path'],
                    'segment_index': result['metadata']['segment_index'],
                    'segment_duration': result['metadata']['segment_duration'],
                    'tempo': result['metadata']['tempo'],
                    'num_beats': result['metadata']['num_beats'],
                    'file_duration': result['metadata']['file_duration']
                }
            }
            formatted_results.append(formatted_result)
        
        return SearchResponse(
            success=True,
            query_file_id=file_id,
            results=formatted_results,
            total_results=len(formatted_results),
            message=f"Found {len(formatted_results)} similar music pieces"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching by ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/files")
async def list_files():
    """
    List all files in the database with their metadata
    
    Returns:
        List of files with metadata
    """
    try:
        # Get collection stats
        stats = pipeline.vector_store.get_collection_stats()
        
        # Get all embeddings (this might be expensive for large collections)
        # For now, return basic stats
        return {
            "success": True,
            "total_files": stats.get("total_embeddings", 0),
            "embedding_dimension": stats.get("embedding_dimension", 0),
            "message": "Use /files/{file_id} to get specific file details"
        }
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    """
    Get details of a specific file by ID
    
    Args:
        file_id: ID of the file in the database
        
    Returns:
        File details with metadata
    """
    try:
        embedding_data = pipeline.vector_store.get_embedding_by_id(file_id)
        
        if not embedding_data:
            raise HTTPException(
                status_code=404, 
                detail=f"File with ID {file_id} not found"
            )
        
        return {
            "success": True,
            "file_id": file_id,
            "metadata": embedding_data['metadata'],
            "message": "File details retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get file: {str(e)}")

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Delete a file from the database
    
    Args:
        file_id: ID of the file to delete
        
    Returns:
        Deletion confirmation
    """
    try:
        success = pipeline.vector_store.delete_embedding(file_id)
        
        if not success:
            raise HTTPException(
                status_code=404, 
                detail=f"File with ID {file_id} not found or could not be deleted"
            )
        
        return {
            "success": True,
            "file_id": file_id,
            "message": "File deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 