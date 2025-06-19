import os
from pathlib import Path
from typing import List

class Config:
    # Audio processing settings
    SAMPLE_RATE = 22050
    DURATION = 30  # seconds to process per file
    HOP_LENGTH = 512
    N_MELS = 128
    
    # Model settings
    MODEL_NAME = "facebook/wav2vec2-base"  # Can be changed to music-specific models
    EMBEDDING_DIM = 768
    
    # Vector database settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    COLLECTION_NAME = "music_embeddings"
    
    # ChromaDB connection settings (for Docker deployment)
    CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    
    # File processing settings
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    BATCH_SIZE = 8
    
    # Paths
    MUSIC_FILES_PATH = os.getenv("MUSIC_FILES_PATH", "./music_files")
    PROCESSED_FILES_PATH = "./processed_files"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.VECTOR_DB_PATH,
            cls.MUSIC_FILES_PATH,
            cls.PROCESSED_FILES_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True) 