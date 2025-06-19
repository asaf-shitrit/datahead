#!/usr/bin/env python3
"""
Main entry point for the Music Embedding Pipeline
Provides options to run the FastAPI server or MCP server
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_api_server():
    """Run the FastAPI server"""
    import uvicorn
    from utils.start_api import main as start_api
    start_api()

def run_mcp_server():
    """Run the MCP server"""
    import asyncio
    from api.mcp_server import main as start_mcp
    asyncio.run(start_mcp())

def run_pipeline():
    """Run the pipeline directly"""
    from scripts.main import main as run_main
    run_main()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Music Embedding Pipeline')
    parser.add_argument(
        '--mode',
        choices=['api', 'mcp', 'pipeline'],
        default='api',
        help='Mode to run: api (FastAPI server), mcp (MCP server), or pipeline (direct pipeline)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (for API server)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (for API server)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        print("🚀 Starting FastAPI server...")
        run_api_server()
    elif args.mode == 'mcp':
        print("🤖 Starting MCP server...")
        run_mcp_server()
    elif args.mode == 'pipeline':
        print("🎵 Running pipeline directly...")
        run_pipeline()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 