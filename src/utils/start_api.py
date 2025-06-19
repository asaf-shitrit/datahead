#!/usr/bin/env python3
"""
Startup script for the Music Embedding API
"""

import uvicorn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server"""
    try:
        logger.info("Starting Music Embedding API server...")
        
        # Run the server
        uvicorn.run(
            "src.api.api:app",
            host="0.0.0.0",
            port=8080,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    main() 