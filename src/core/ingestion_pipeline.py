import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
import time

from ..utils.config import Config
from .audio_processor import AudioProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicIngestionPipeline:
    """Main pipeline for ingesting music files and generating embeddings"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
        # Create necessary directories
        Config.create_directories()
        
        logger.info("Music ingestion pipeline initialized")
    
    def find_music_files(self, directory: str = None) -> List[str]:
        """
        Find all supported music files in the specified directory
        
        Args:
            directory: Directory to search for music files
            
        Returns:
            List of file paths
        """
        search_dir = directory or Config.MUSIC_FILES_PATH
        music_files = []
        
        try:
            for file_path in Path(search_dir).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in Config.SUPPORTED_FORMATS:
                    music_files.append(str(file_path))
            
            logger.info(f"Found {len(music_files)} music files in {search_dir}")
            return music_files
            
        except Exception as e:
            logger.error(f"Error finding music files: {str(e)}")
            return []
    
    def process_single_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single music file and generate embeddings
        
        Args:
            file_path: Path to the music file
            
        Returns:
            Dictionary containing embeddings and metadata, or None if failed
        """
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Load and process audio
            audio_data, sample_rate = self.audio_processor.load_audio(file_path)
            
            # Extract metadata
            metadata = self.audio_processor.get_audio_metadata(file_path)
            
            # Segment audio
            audio_segments = self.audio_processor.segment_audio(audio_data)
            
            if not audio_segments:
                logger.warning(f"No valid segments found in {file_path}")
                return None
            
            # Extract features for each segment
            feature_segments = []
            for segment in audio_segments:
                features = self.audio_processor.extract_features(segment)
                feature_segments.append(features)
            
            # Generate embeddings for each segment
            embeddings = self.embedding_generator.batch_generate_embeddings(
                audio_segments, feature_segments
            )
            
            # Prepare results
            results = {
                'file_path': file_path,
                'metadata': metadata,
                'segments': []
            }
            
            for i, (embedding, segment_features) in enumerate(zip(embeddings, feature_segments)):
                segment_data = {
                    'segment_id': f"{Path(file_path).stem}_segment_{i}",
                    'embedding': embedding,
                    'segment_metadata': {
                        'segment_index': i,
                        'segment_duration': len(audio_segments[i]) / sample_rate,
                        'tempo': segment_features.get('tempo', 0),
                        'num_beats': len(segment_features.get('beats', []))
                    }
                }
                results['segments'].append(segment_data)
            
            logger.info(f"Successfully processed {file_path} into {len(embeddings)} segments")
            return results
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def save_to_vector_store(self, processed_results: List[Dict[str, Any]]) -> bool:
        """
        Save processed results to the vector store
        
        Args:
            processed_results: List of processed file results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            all_embeddings = []
            all_metadata = []
            all_ids = []
            
            for result in processed_results:
                file_path = result['file_path']
                file_metadata = result['metadata']
                
                for segment in result['segments']:
                    # Create unique ID
                    segment_id = f"{uuid.uuid4()}"
                    
                    # Prepare metadata
                    metadata = {
                        'file_path': file_path,
                        'filename': Path(file_path).name,
                        'segment_id': segment['segment_id'],
                        'segment_index': segment['segment_metadata']['segment_index'],
                        'segment_duration': segment['segment_metadata']['segment_duration'],
                        'tempo': segment['segment_metadata']['tempo'],
                        'num_beats': segment['segment_metadata']['num_beats'],
                        'file_duration': file_metadata.get('duration', 0),
                        'file_size': file_metadata.get('file_size', 0),
                        'file_extension': file_metadata.get('file_extension', ''),
                        'processed_timestamp': time.time()
                    }
                    
                    all_embeddings.append(segment['embedding'])
                    all_metadata.append(metadata)
                    all_ids.append(segment_id)
            
            # Save to vector store
            success = self.vector_store.add_embeddings(all_embeddings, all_metadata, all_ids)
            
            if success:
                logger.info(f"Successfully saved {len(all_embeddings)} embeddings to vector store")
            else:
                logger.error("Failed to save embeddings to vector store")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving to vector store: {str(e)}")
            return False
    
    def run_pipeline(self, music_directory: str = None, batch_size: int = None) -> Dict[str, Any]:
        """
        Run the complete ingestion pipeline
        
        Args:
            music_directory: Directory containing music files
            batch_size: Number of files to process in each batch
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        
        # Find music files
        music_files = self.find_music_files(music_directory)
        
        if not music_files:
            logger.warning("No music files found")
            return {'success': False, 'error': 'No music files found'}
        
        # Process files
        batch_size = batch_size or Config.BATCH_SIZE
        processed_results = []
        failed_files = []
        
        logger.info(f"Starting pipeline with {len(music_files)} files")
        
        for i in tqdm(range(0, len(music_files), batch_size), desc="Processing batches"):
            batch_files = music_files[i:i + batch_size]
            
            for file_path in batch_files:
                try:
                    result = self.process_single_file(file_path)
                    if result:
                        processed_results.append(result)
                    else:
                        failed_files.append(file_path)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    failed_files.append(file_path)
        
        # Save to vector store
        if processed_results:
            save_success = self.save_to_vector_store(processed_results)
        else:
            save_success = False
        
        # Calculate statistics
        total_segments = sum(len(result['segments']) for result in processed_results)
        processing_time = time.time() - start_time
        
        # Get vector store stats
        vector_stats = self.vector_store.get_collection_stats()
        
        results = {
            'success': save_success,
            'total_files': len(music_files),
            'processed_files': len(processed_results),
            'failed_files': len(failed_files),
            'total_segments': total_segments,
            'processing_time_seconds': processing_time,
            'vector_store_stats': vector_stats,
            'failed_file_paths': failed_files
        }
        
        logger.info(f"Pipeline completed: {results}")
        return results
    
    def search_similar_music(self, query_file_path: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar music using a query file
        
        Args:
            query_file_path: Path to the query music file
            n_results: Number of similar results to return
            
        Returns:
            List of similar music results
        """
        try:
            # Process the query file
            query_result = self.process_single_file(query_file_path)
            
            if not query_result or not query_result['segments']:
                logger.error("Could not process query file")
                return []
            
            # Use the first segment as query
            query_embedding = query_result['segments'][0]['embedding']
            
            # Search for similar embeddings
            similar_results = self.vector_store.search_similar(
                query_embedding, n_results=n_results
            )
            
            logger.info(f"Found {len(similar_results)} similar music pieces")
            return similar_results
            
        except Exception as e:
            logger.error(f"Error searching similar music: {str(e)}")
            return []
    
    def export_pipeline_results(self, output_file: str) -> bool:
        """
        Export pipeline results and metadata
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.vector_store.export_metadata(output_file)
        except Exception as e:
            logger.error(f"Error exporting pipeline results: {str(e)}")
            return False 