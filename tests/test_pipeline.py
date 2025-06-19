#!/usr/bin/env python3
"""
Test script for the Music Embedding Pipeline
Tests the core functionality of the pipeline
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging
import unittest
from unittest.mock import Mock, patch
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.ingestion_pipeline import MusicIngestionPipeline
from src.core.audio_processor import AudioProcessor
from src.core.embedding_generator import EmbeddingGenerator
from src.core.vector_store import VectorStore
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        print(f"✓ Sample rate: {Config.SAMPLE_RATE}")
        print(f"✓ Duration: {Config.DURATION}")
        print(f"✓ Model: {Config.MODEL_NAME}")
        print(f"✓ Supported formats: {Config.SUPPORTED_FORMATS}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_audio_processor():
    """Test audio processor"""
    print("\nTesting audio processor...")
    try:
        processor = AudioProcessor()
        print("✓ Audio processor initialized")
        
        # Test with a dummy audio array
        dummy_audio = np.random.randn(22050)  # 1 second of random audio
        features = processor.extract_features(dummy_audio)
        print(f"✓ Feature extraction successful: {list(features.keys())}")
        
        segments = processor.segment_audio(dummy_audio)
        print(f"✓ Audio segmentation successful: {len(segments)} segments")
        
        return True
    except Exception as e:
        print(f"✗ Audio processor test failed: {e}")
        return False

def test_embedding_generator():
    """Test embedding generator"""
    print("\nTesting embedding generator...")
    try:
        generator = EmbeddingGenerator()
        print("✓ Embedding generator initialized")
        
        # Test with dummy data
        dummy_audio = np.random.randn(22050)
        dummy_features = {
            'mel_spectrogram': np.random.randn(128, 100),
            'mfcc': np.random.randn(13, 100),
            'spectral_centroid': np.random.randn(100),
            'tempo': 120.0,
            'chroma': np.random.randn(12, 100)
        }
        
        # Test feature embedding
        feature_embedding = generator.generate_feature_embedding(dummy_features)
        print(f"✓ Feature embedding generated: {feature_embedding.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding generator test failed: {e}")
        return False

def test_vector_store():
    """Test vector store"""
    print("\nTesting vector store...")
    try:
        # Use a temporary directory for testing
        test_db_path = "./test_vector_db"
        store = VectorStore(persist_directory=test_db_path)
        print("✓ Vector store initialized")
        
        # Test adding dummy embeddings
        dummy_embeddings = [np.random.randn(768) for _ in range(3)]
        dummy_metadata = [{"test": f"data_{i}"} for i in range(3)]
        dummy_ids = [f"test_id_{i}" for i in range(3)]
        
        success = store.add_embeddings(dummy_embeddings, dummy_metadata, dummy_ids)
        if success:
            print("✓ Embeddings added successfully")
        else:
            print("✗ Failed to add embeddings")
            return False
        
        # Test search
        query_embedding = np.random.randn(768)
        results = store.search_similar(query_embedding, n_results=2)
        print(f"✓ Search successful: {len(results)} results")
        
        # Test stats
        stats = store.get_collection_stats()
        print(f"✓ Collection stats: {stats}")
        
        # Clean up
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
        
        return True
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\nTesting pipeline initialization...")
    try:
        pipeline = MusicIngestionPipeline()
        print("✓ Pipeline initialized successfully")
        
        # Test finding music files (should work even if directory is empty)
        music_files = pipeline.find_music_files("./music_files")
        print(f"✓ Music file discovery: {len(music_files)} files found")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    required_packages = [
        'librosa', 'numpy', 'pandas', 'sklearn', 'torch', 
        'transformers', 'chromadb', 'pydub', 'soundfile', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Music Embedding Pipeline - Component Tests")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_config),
        ("Audio Processor", test_audio_processor),
        ("Embedding Generator", test_embedding_generator),
        ("Vector Store", test_vector_store),
        ("Pipeline Initialization", test_pipeline_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Add music files to the ./music_files directory")
        print("2. Run: python main.py")
        print("3. Or run: python example_usage.py for examples")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check if you have sufficient disk space")
        print("3. Ensure you have internet connection for downloading models")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 