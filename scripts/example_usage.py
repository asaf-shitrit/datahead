#!/usr/bin/env python3
"""
Example usage of the Music Embedding Pipeline
Demonstrates how to use the pipeline for processing music files
"""

import os
import sys
from pathlib import Path
import logging
from src.core.ingestion_pipeline import MusicIngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Example of basic pipeline usage"""
    print("=== Basic Pipeline Usage ===")
    
    # Initialize the pipeline
    pipeline = MusicIngestionPipeline()
    
    # Check if music files exist
    music_dir = "./music_files"
    if not os.path.exists(music_dir):
        print(f"Music directory {music_dir} not found. Creating it...")
        os.makedirs(music_dir, exist_ok=True)
        print(f"Please add some music files to {music_dir} and run this example again.")
        return
    
    # Find music files
    music_files = pipeline.find_music_files(music_dir)
    print(f"Found {len(music_files)} music files")
    
    if not music_files:
        print("No music files found. Please add some music files to the music_files directory.")
        return
    
    # Process a single file as an example
    print(f"\nProcessing first file: {music_files[0]}")
    result = pipeline.process_single_file(music_files[0])
    
    if result:
        print(f"Successfully processed file with {len(result['segments'])} segments")
        
        # Save to vector store
        success = pipeline.save_to_vector_store([result])
        if success:
            print("Successfully saved to vector store")
        else:
            print("Failed to save to vector store")
    else:
        print("Failed to process file")

def example_search_functionality():
    """Example of search functionality"""
    print("\n=== Search Functionality Example ===")
    
    pipeline = MusicIngestionPipeline()
    
    # Check if we have any embeddings in the vector store
    stats = pipeline.vector_store.get_collection_stats()
    if stats.get('total_embeddings', 0) == 0:
        print("No embeddings found in vector store. Please run the basic example first.")
        return
    
    print(f"Found {stats['total_embeddings']} embeddings in vector store")
    
    # Example: search for similar music (if we have a query file)
    music_dir = "./music_files"
    if os.path.exists(music_dir):
        music_files = pipeline.find_music_files(music_dir)
        if music_files:
            query_file = music_files[0]
            print(f"\nSearching for music similar to: {query_file}")
            
            similar_results = pipeline.search_similar_music(query_file, n_results=5)
            
            if similar_results:
                print(f"Found {len(similar_results)} similar pieces:")
                for i, result in enumerate(similar_results, 1):
                    metadata = result['metadata']
                    print(f"{i}. {metadata['filename']} (Segment {metadata['segment_index']})")
                    print(f"   Tempo: {metadata['tempo']:.1f} BPM")
                    print(f"   Distance: {result['distance']:.4f}")
            else:
                print("No similar music found")

def example_batch_processing():
    """Example of batch processing"""
    print("\n=== Batch Processing Example ===")
    
    pipeline = MusicIngestionPipeline()
    
    # Run the full pipeline
    music_dir = "./music_files"
    if not os.path.exists(music_dir):
        print(f"Music directory {music_dir} not found.")
        return
    
    print("Running full pipeline...")
    results = pipeline.run_pipeline(music_directory=music_dir, batch_size=4)
    
    print(f"Pipeline completed:")
    print(f"  Success: {results['success']}")
    print(f"  Total files: {results['total_files']}")
    print(f"  Processed: {results['processed_files']}")
    print(f"  Failed: {results['failed_files']}")
    print(f"  Total segments: {results['total_segments']}")
    print(f"  Processing time: {results['processing_time_seconds']:.2f} seconds")

def example_export_results():
    """Example of exporting results"""
    print("\n=== Export Results Example ===")
    
    pipeline = MusicIngestionPipeline()
    
    # Export metadata to JSON file
    export_file = "./pipeline_results.json"
    success = pipeline.export_pipeline_results(export_file)
    
    if success:
        print(f"Results exported to: {export_file}")
    else:
        print("Failed to export results")

def example_custom_configuration():
    """Example of custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # You can modify the configuration before initializing the pipeline
    from src.utils.config import Config
    
    # Example: change some settings
    Config.DURATION = 20  # Process 20-second segments instead of 30
    Config.BATCH_SIZE = 4  # Process 4 files at a time
    
    print(f"Modified configuration:")
    print(f"  Segment duration: {Config.DURATION} seconds")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    
    # Initialize pipeline with custom config
    pipeline = MusicIngestionPipeline()
    
    # The pipeline will use the modified configuration
    print("Pipeline initialized with custom configuration")

def main():
    """Run all examples"""
    print("Music Embedding Pipeline - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_search_functionality()
        example_batch_processing()
        example_export_results()
        example_custom_configuration()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        print("\nTo use the pipeline with your own music files:")
        print("1. Add music files to the ./music_files directory")
        print("2. Run: python main.py")
        print("3. Or use the pipeline programmatically as shown in these examples")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 