#!/usr/bin/env python3
"""
Setup script for the Music Embedding Ingestion Pipeline
This script helps with initial setup and installation
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "music_files",
        "vector_db", 
        "processed_files"
    ]
    
    print("Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created: {directory}")

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run the test suite"""
    print("Running tests...")
    
    try:
        subprocess.check_call([sys.executable, "test_pipeline.py"])
        print("✓ All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Music Embedding Pipeline - Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\nFailed to install dependencies. Please try manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("\nTests failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your music files to the 'music_files' directory")
    print("2. Run the pipeline: python main.py")
    print("3. Or try the examples: python example_usage.py")
    print("\nSupported audio formats: MP3, WAV, FLAC, M4A, OGG")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 