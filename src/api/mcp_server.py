#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for Music Embedding API
Provides tools for LLMs to interact with music similarity search
"""

import asyncio
import json
import logging
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    LogMessage,
    PublishDiagnosticsParams,
    Diagnostic,
    DiagnosticSeverity,
    Range,
    Position,
)
import mcp.server as mcp_server

from ..core.ingestion_pipeline import MusicIngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("music-embedding-mcp")

# Initialize the music pipeline
pipeline = MusicIngestionPipeline()

@dataclass
class MusicSearchResult:
    """Represents a music search result"""
    id: str
    filename: str
    distance: float
    tempo: float
    duration: float
    segment_index: int

class MusicEmbeddingMCPServer:
    """MCP Server for Music Embedding functionality"""
    
    def __init__(self):
        self.pipeline = pipeline
        self.temp_files = {}  # Track temporary files for cleanup
    
    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List available tools"""
        tools = [
            Tool(
                name="upload_music_file",
                description="Upload a music file and generate embeddings for similarity search. Supports MP3, WAV, FLAC, M4A, AAC, OGG, WMA formats.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the music file to upload"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description or notes about the music file"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="search_similar_music",
                description="Search for similar music using a query file. Returns a list of similar music pieces with metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query_file_path": {
                            "type": "string",
                            "description": "Path to the query music file"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of similar results to return (1-50)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["query_file_path"]
                }
            ),
            Tool(
                name="search_by_file_id",
                description="Search for similar music using a file ID from the database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "ID of the file in the database to search for similar music"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of similar results to return (1-50)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["file_id"]
                }
            ),
            Tool(
                name="get_file_info",
                description="Get detailed information about a specific file by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "ID of the file in the database"
                        }
                    },
                    "required": ["file_id"]
                }
            ),
            Tool(
                name="list_all_files",
                description="Get a list of all files in the database with basic statistics.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="delete_file",
                description="Delete a file from the database by ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "ID of the file to delete"
                        }
                    },
                    "required": ["file_id"]
                }
            )
        ]
        
        return ListToolsResult(tools=tools)
    
    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls"""
        try:
            tool_name = request.name
            arguments = request.arguments
            
            logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
            
            if tool_name == "upload_music_file":
                return await self._upload_music_file(arguments)
            elif tool_name == "search_similar_music":
                return await self._search_similar_music(arguments)
            elif tool_name == "search_by_file_id":
                return await self._search_by_file_id(arguments)
            elif tool_name == "get_file_info":
                return await self._get_file_info(arguments)
            elif tool_name == "list_all_files":
                return await self._list_all_files(arguments)
            elif tool_name == "delete_file":
                return await self._delete_file(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error in tool call {request.name}: {str(e)}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: {str(e)}"
                    )
                ],
                isError=True
            )
    
    async def _upload_music_file(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Upload and process a music file"""
        file_path = arguments.get("file_path")
        description = arguments.get("description", "")
        
        if not os.path.exists(file_path):
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: File not found: {file_path}")],
                isError=True
            )
        
        try:
            # Process the file
            result = self.pipeline.process_single_file(file_path)
            
            if not result:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Failed to process file: {file_path}")],
                    isError=True
                )
            
            # Save to vector store
            save_success = self.pipeline.save_to_vector_store([result])
            
            if not save_success:
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: Failed to save embeddings to vector store")],
                    isError=True
                )
            
            # Prepare response
            file_id = result['segments'][0]['segment_id'] if result['segments'] else None
            segments_info = []
            
            for seg in result['segments']:
                segments_info.append({
                    'segment_id': seg['segment_id'],
                    'duration': seg['segment_metadata']['segment_duration'],
                    'tempo': seg['segment_metadata']['tempo']
                })
            
            response_text = f"""✅ Successfully uploaded and processed: {os.path.basename(file_path)}

📊 File Information:
- File ID: {file_id}
- Segments Created: {len(result['segments'])}
- File Size: {result['metadata'].get('file_size', 0)} bytes
- Duration: {result['metadata'].get('duration', 0):.2f} seconds
- Description: {description}

🎵 Segments:
{json.dumps(segments_info, indent=2)}

The file is now available for similarity search!"""
            
            return CallToolResult(
                content=[TextContent(type="text", text=response_text)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error processing file: {str(e)}")],
                isError=True
            )
    
    async def _search_similar_music(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Search for similar music using a query file"""
        query_file_path = arguments.get("query_file_path")
        n_results = arguments.get("n_results", 10)
        
        if not os.path.exists(query_file_path):
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: Query file not found: {query_file_path}")],
                isError=True
            )
        
        try:
            # Search for similar music
            similar_results = self.pipeline.search_similar_music(query_file_path, n_results=n_results)
            
            if not similar_results:
                return CallToolResult(
                    content=[TextContent(type="text", text="No similar music found in the database.")]
                )
            
            # Format results
            results_text = f"🎵 Found {len(similar_results)} similar music pieces for: {os.path.basename(query_file_path)}\n\n"
            
            for i, result in enumerate(similar_results, 1):
                metadata = result['metadata']
                results_text += f"{i}. **{metadata['filename']}** (Segment {metadata['segment_index']})\n"
                results_text += f"   - Similarity Score: {1 - result['distance']:.4f}\n"
                results_text += f"   - Tempo: {metadata['tempo']:.1f} BPM\n"
                results_text += f"   - Duration: {metadata['segment_duration']:.1f}s\n"
                results_text += f"   - File ID: {result['id']}\n\n"
            
            return CallToolResult(
                content=[TextContent(type="text", text=results_text)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error searching similar music: {str(e)}")],
                isError=True
            )
    
    async def _search_by_file_id(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Search for similar music using a file ID"""
        file_id = arguments.get("file_id")
        n_results = arguments.get("n_results", 10)
        
        try:
            # Get embedding by ID
            embedding_data = self.pipeline.vector_store.get_embedding_by_id(file_id)
            
            if not embedding_data:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: File with ID {file_id} not found in database")],
                    isError=True
                )
            
            # Search for similar embeddings
            similar_results = self.pipeline.vector_store.search_similar(
                embedding_data['embedding'], n_results=n_results
            )
            
            if not similar_results:
                return CallToolResult(
                    content=[TextContent(type="text", text="No similar music found.")]
                )
            
            # Format results
            query_filename = embedding_data['metadata']['filename']
            results_text = f"🎵 Found {len(similar_results)} similar music pieces for: {query_filename}\n\n"
            
            for i, result in enumerate(similar_results, 1):
                metadata = result['metadata']
                results_text += f"{i}. **{metadata['filename']}** (Segment {metadata['segment_index']})\n"
                results_text += f"   - Similarity Score: {1 - result['distance']:.4f}\n"
                results_text += f"   - Tempo: {metadata['tempo']:.1f} BPM\n"
                results_text += f"   - Duration: {metadata['segment_duration']:.1f}s\n"
                results_text += f"   - File ID: {result['id']}\n\n"
            
            return CallToolResult(
                content=[TextContent(type="text", text=results_text)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error searching by file ID: {str(e)}")],
                isError=True
            )
    
    async def _get_file_info(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get detailed information about a file"""
        file_id = arguments.get("file_id")
        
        try:
            embedding_data = self.pipeline.vector_store.get_embedding_by_id(file_id)
            
            if not embedding_data:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: File with ID {file_id} not found")],
                    isError=True
                )
            
            metadata = embedding_data['metadata']
            info_text = f"""📁 File Information for ID: {file_id}

📄 Basic Info:
- Filename: {metadata['filename']}
- File Path: {metadata['file_path']}
- File Extension: {metadata['file_extension']}
- File Size: {metadata['file_size']} bytes
- File Duration: {metadata['file_duration']:.2f} seconds

🎵 Segment Info:
- Segment ID: {metadata['segment_id']}
- Segment Index: {metadata['segment_index']}
- Segment Duration: {metadata['segment_duration']:.2f} seconds
- Tempo: {metadata['tempo']:.1f} BPM
- Number of Beats: {metadata['num_beats']}

⏰ Processing:
- Processed Timestamp: {metadata['processed_timestamp']}"""
            
            return CallToolResult(
                content=[TextContent(type="text", text=info_text)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error getting file info: {str(e)}")],
                isError=True
            )
    
    async def _list_all_files(self, arguments: Dict[str, Any]) -> CallToolResult:
        """List all files in the database"""
        try:
            stats = self.pipeline.vector_store.get_collection_stats()
            
            info_text = f"""📊 Database Statistics

📈 Collection Info:
- Total Embeddings: {stats.get('total_embeddings', 0)}
- Embedding Dimension: {stats.get('embedding_dimension', 0)}
- Collection Name: {stats.get('collection_name', 'music_embeddings')}

💾 Storage:
- Persist Directory: {stats.get('persist_directory', 'N/A')}

ℹ️ Note: Use the 'get_file_info' tool with a specific file ID to get detailed information about individual files."""
            
            return CallToolResult(
                content=[TextContent(type="text", text=info_text)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error listing files: {str(e)}")],
                isError=True
            )
    
    async def _delete_file(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Delete a file from the database"""
        file_id = arguments.get("file_id")
        
        try:
            success = self.pipeline.vector_store.delete_embedding(file_id)
            
            if not success:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: File with ID {file_id} not found or could not be deleted")],
                    isError=True
                )
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"✅ Successfully deleted file with ID: {file_id}")]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error deleting file: {str(e)}")],
                isError=True
            )

# Initialize the MCP server instance
mcp_server_instance = MusicEmbeddingMCPServer()

@server.list_tools()
async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
    """Handle list tools request"""
    return await mcp_server_instance.list_tools(request)

@server.call_tool()
async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool call request"""
    return await mcp_server_instance.call_tool(request)

async def main():
    """Main function to run the MCP server"""
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="music-embedding-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 