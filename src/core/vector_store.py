import chromadb
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector database operations using ChromaDB"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or Config.VECTOR_DB_PATH
        self.collection_name = Config.COLLECTION_NAME
        self.client = None
        self.collection = None
        
        # Check if we should use remote ChromaDB
        self.use_remote = Config.CHROMA_HOST != "localhost"
        
        if self.use_remote:
            logger.info(f"Initializing remote vector store at: {Config.CHROMA_HOST}:{Config.CHROMA_PORT}")
        else:
            logger.info(f"Initializing local vector store at: {self.persist_directory}")
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            if self.use_remote:
                # Connect to remote ChromaDB server
                self.client = chromadb.HttpClient(
                    host=Config.CHROMA_HOST,
                    port=Config.CHROMA_PORT
                )
            else:
                # Create persist directory if it doesn't exist
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                
                # Initialize local ChromaDB client
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Music embeddings collection"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_embeddings(self, embeddings: List[np.ndarray], 
                      metadata: List[Dict[str, Any]], 
                      ids: List[str]) -> bool:
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            ids: List of unique identifiers
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy arrays to lists for ChromaDB
            embedding_list = [embedding.tolist() for embedding in embeddings]
            
            # Add embeddings to collection
            self.collection.add(
                embeddings=embedding_list,
                metadatas=metadata,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(embeddings)} embeddings to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to vector store: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, 
                      n_results: int = 10, 
                      filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar results with metadata
        """
        try:
            # Convert numpy array to list
            query_list = query_embedding.tolist()
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar embeddings")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve embedding by ID
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            Embedding data with metadata or None if not found
        """
        try:
            results = self.collection.get(ids=[embedding_id])
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'embedding': np.array(results['embeddings'][0]),
                    'metadata': results['metadatas'][0]
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving embedding by ID: {str(e)}")
            return None
    
    def update_embedding(self, embedding_id: str, 
                        new_embedding: np.ndarray, 
                        new_metadata: Dict[str, Any]) -> bool:
        """
        Update an existing embedding
        
        Args:
            embedding_id: Unique identifier
            new_embedding: New embedding vector
            new_metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.update(
                ids=[embedding_id],
                embeddings=[new_embedding.tolist()],
                metadatas=[new_metadata]
            )
            
            logger.info(f"Successfully updated embedding: {embedding_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating embedding: {str(e)}")
            return False
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding by ID
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[embedding_id])
            logger.info(f"Successfully deleted embedding: {embedding_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embedding: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample embeddings to determine dimension
            sample_results = self.collection.peek(limit=1)
            embedding_dim = len(sample_results['embeddings'][0]) if sample_results['embeddings'] else 0
            
            stats = {
                'total_embeddings': count,
                'embedding_dimension': embedding_dim,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def export_metadata(self, output_file: str) -> bool:
        """
        Export all metadata to a JSON file
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all embeddings
            results = self.collection.get()
            
            # Prepare export data
            export_data = {
                'collection_name': self.collection_name,
                'total_embeddings': len(results['ids']),
                'embeddings': []
            }
            
            for i in range(len(results['ids'])):
                embedding_data = {
                    'id': results['ids'][i],
                    'metadata': results['metadatas'][i]
                }
                export_data['embeddings'].append(embedding_data)
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported metadata to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metadata: {str(e)}")
            return False 