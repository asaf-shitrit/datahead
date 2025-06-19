#!/usr/bin/env python3
"""
Music Embedding Ingestion Pipeline
Main script for processing music files and generating embeddings
"""

import argparse
import sys
import os
from pathlib import Path
import logging
from src.core.ingestion_pipeline import MusicIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Music Embedding Ingestion Pipeline')
    parser.add_argument(
        '--music-dir', 
        type=str, 
        default='./music_files',
        help='Directory containing music files to process'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=8,
        help='Number of files to process in each batch'
    )
    parser.add_argument(
        '--search-query', 
        type=str,
        help='Path to a music file to search for similar music'
    )
    parser.add_argument(
        '--n-results', 
        type=int, 
        default=10,
        help='Number of similar results to return when searching'
    )
    parser.add_argument(
        '--export-results', 
        type=str,
        help='Export pipeline results to specified JSON file'
    )
    parser.add_argument(
        '--process-single', 
        type=str,
        help='Process a single music file instead of a directory'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = MusicIngestionPipeline()
        
        if args.process_single:
            # Process a single file
            if not os.path.exists(args.process_single):
                logger.error(f"File not found: {args.process_single}")
                sys.exit(1)
            
            logger.info(f"Processing single file: {args.process_single}")
            result = pipeline.process_single_file(args.process_single)
            
            if result:
                logger.info(f"Successfully processed file with {len(result['segments'])} segments")
                # Save to vector store
                pipeline.save_to_vector_store([result])
            else:
                logger.error("Failed to process file")
                sys.exit(1)
        
        elif args.search_query:
            # Search for similar music
            if not os.path.exists(args.search_query):
                logger.error(f"Query file not found: {args.search_query}")
                sys.exit(1)
            
            logger.info(f"Searching for music similar to: {args.search_query}")
            similar_results = pipeline.search_similar_music(
                args.search_query, n_results=args.n_results
            )
            
            if similar_results:
                print(f"\nFound {len(similar_results)} similar music pieces:")
                for i, result in enumerate(similar_results, 1):
                    metadata = result['metadata']
                    print(f"{i}. {metadata['filename']} (Segment {metadata['segment_index']})")
                    print(f"   Tempo: {metadata['tempo']:.1f} BPM")
                    print(f"   Duration: {metadata['segment_duration']:.1f}s")
                    print(f"   Distance: {result['distance']:.4f}")
                    print()
            else:
                print("No similar music found")
        
        else:
            # Run full pipeline
            music_dir = args.music_dir
            
            if not os.path.exists(music_dir):
                logger.error(f"Music directory not found: {music_dir}")
                logger.info("Creating music directory...")
                os.makedirs(music_dir, exist_ok=True)
                logger.info(f"Please add music files to: {music_dir}")
                sys.exit(1)
            
            logger.info(f"Starting music ingestion pipeline...")
            logger.info(f"Music directory: {music_dir}")
            logger.info(f"Batch size: {args.batch_size}")
            
            # Run pipeline
            results = pipeline.run_pipeline(
                music_directory=music_dir,
                batch_size=args.batch_size
            )
            
            # Print results
            print("\n" + "="*50)
            print("PIPELINE RESULTS")
            print("="*50)
            print(f"Success: {results['success']}")
            print(f"Total files found: {results['total_files']}")
            print(f"Files processed: {results['processed_files']}")
            print(f"Files failed: {results['failed_files']}")
            print(f"Total segments created: {results['total_segments']}")
            print(f"Processing time: {results['processing_time_seconds']:.2f} seconds")
            
            if results['vector_store_stats']:
                print(f"Vector store embeddings: {results['vector_store_stats'].get('total_embeddings', 0)}")
                print(f"Embedding dimension: {results['vector_store_stats'].get('embedding_dimension', 0)}")
            
            if results['failed_file_paths']:
                print(f"\nFailed files:")
                for failed_file in results['failed_file_paths']:
                    print(f"  - {failed_file}")
            
            # Export results if requested
            if args.export_results:
                logger.info(f"Exporting results to: {args.export_results}")
                export_success = pipeline.export_pipeline_results(args.export_results)
                if export_success:
                    print(f"Results exported to: {args.export_results}")
                else:
                    print("Failed to export results")
        
        print("\nPipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 