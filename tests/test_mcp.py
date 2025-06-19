#!/usr/bin/env python3
"""
Test script for the Music Embedding MCP Server
Demonstrates how to use the MCP server programmatically
"""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_server():
    """Test the MCP server functionality"""
    
    # This is a simplified test - in practice, you'd use an MCP client
    # to communicate with the server via stdio
    
    print("🎵 Music Embedding MCP Server Test")
    print("=" * 50)
    
    # Simulate what tools would be available
    tools = [
        {
            "name": "upload_music_file",
            "description": "Upload a music file and generate embeddings for similarity search",
            "parameters": {
                "file_path": "Path to the music file",
                "description": "Optional description"
            }
        },
        {
            "name": "search_similar_music", 
            "description": "Search for similar music using a query file",
            "parameters": {
                "query_file_path": "Path to the query music file",
                "n_results": "Number of results (1-50)"
            }
        },
        {
            "name": "search_by_file_id",
            "description": "Search for similar music using a file ID",
            "parameters": {
                "file_id": "ID of the file in database",
                "n_results": "Number of results (1-50)"
            }
        },
        {
            "name": "get_file_info",
            "description": "Get detailed information about a file",
            "parameters": {
                "file_id": "ID of the file in database"
            }
        },
        {
            "name": "list_all_files",
            "description": "List all files in the database",
            "parameters": {}
        },
        {
            "name": "delete_file",
            "description": "Delete a file from the database",
            "parameters": {
                "file_id": "ID of the file to delete"
            }
        }
    ]
    
    print("📋 Available Tools:")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool['name']}")
        print(f"   Description: {tool['description']}")
        if tool['parameters']:
            print(f"   Parameters: {', '.join(tool['parameters'].keys())}")
        print()
    
    print("🔧 Example Usage Scenarios:")
    print()
    
    # Example 1: Upload a music file
    print("1. Upload a Music File:")
    print("   Tool: upload_music_file")
    print("   Arguments: {'file_path': '/path/to/song.mp3', 'description': 'My favorite song'}")
    print("   Expected Result: File processed and embeddings stored in database")
    print()
    
    # Example 2: Search for similar music
    print("2. Search for Similar Music:")
    print("   Tool: search_similar_music")
    print("   Arguments: {'query_file_path': '/path/to/query.mp3', 'n_results': 5}")
    print("   Expected Result: List of similar music pieces with similarity scores")
    print()
    
    # Example 3: Get file information
    print("3. Get File Information:")
    print("   Tool: get_file_info")
    print("   Arguments: {'file_id': 'song_segment_0'}")
    print("   Expected Result: Detailed metadata about the file")
    print()
    
    print("🚀 To use with an MCP client:")
    print("1. Start the MCP server: python mcp_server.py")
    print("2. Configure your MCP client with mcp_config.json")
    print("3. The LLM can now use these tools for music similarity search!")
    print()
    
    print("💡 LLM Integration Examples:")
    print()
    print("User: 'I have a song called song.mp3, can you find similar music?'")
    print("LLM: I'll help you find similar music! Let me first upload your song and then search for similar pieces.")
    print("     [Calls upload_music_file with song.mp3]")
    print("     [Calls search_similar_music with the uploaded file]")
    print()
    
    print("User: 'What's the tempo of the song with ID song_segment_0?'")
    print("LLM: Let me get the detailed information about that song for you.")
    print("     [Calls get_file_info with song_segment_0]")
    print()
    
    print("User: 'Show me all the music in the database'")
    print("LLM: I'll show you an overview of all the music files in the database.")
    print("     [Calls list_all_files]")

if __name__ == "__main__":
    asyncio.run(test_mcp_server()) 