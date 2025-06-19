# Music Embedding Ingestion Pipeline

A comprehensive Python pipeline for processing music files, extracting audio features, generating embeddings, and storing them in a vector database for similarity search and music recommendation.

## Features

- **Multi-format Audio Support**: Handles MP3, WAV, FLAC, M4A, and OGG files
- **Advanced Audio Processing**: Extracts mel spectrograms, MFCC, spectral features, tempo, beats, and chroma features
- **Transformer-based Embeddings**: Uses Wav2Vec2 model for high-quality audio embeddings
- **Vector Database Integration**: Stores embeddings in ChromaDB for efficient similarity search
- **Batch Processing**: Process multiple files efficiently with configurable batch sizes
- **Music Similarity Search**: Find similar music pieces using embedding similarity
- **Comprehensive Metadata**: Stores rich metadata including tempo, duration, beats, and file information
- **Flexible Configuration**: Easy to customize processing parameters
- **Export Capabilities**: Export results and metadata to JSON format

## Architecture

The pipeline consists of four main components:

1. **AudioProcessor**: Handles audio file loading, preprocessing, and feature extraction
2. **EmbeddingGenerator**: Generates embeddings using transformer models and audio features
3. **VectorStore**: Manages vector database operations using ChromaDB
4. **MusicIngestionPipeline**: Orchestrates the entire process

## Installation

1. Clone the repository and navigate to the `generate-embeddings` directory:
```bash
cd generate-embeddings
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create the necessary directories:
```bash
mkdir music_files vector_db processed_files
```

## Quick Start

### 1. Add Music Files

Place your music files in the `music_files` directory. Supported formats: MP3, WAV, FLAC, M4A, OGG.

### 2. Run the Pipeline

Process all music files in the directory:
```bash
python main.py
```

Process a specific directory:
```bash
python main.py --music-dir /path/to/your/music
```

Process a single file:
```bash
python main.py --process-single /path/to/song.mp3
```

### 3. Search for Similar Music

Find music similar to a specific file:
```bash
python main.py --search-query /path/to/query_song.mp3 --n-results 10
```

### 4. Export Results

Export pipeline results to JSON:
```bash
python main.py --export-results results.json
```

## Configuration

The pipeline can be configured by modifying the `config.py` file:

```python
class Config:
    # Audio processing settings
    SAMPLE_RATE = 22050
    DURATION = 30  # seconds to process per file
    HOP_LENGTH = 512
    N_MELS = 128
    
    # Model settings
    MODEL_NAME = "facebook/wav2vec2-base"
    EMBEDDING_DIM = 768
    
    # Vector database settings
    VECTOR_DB_PATH = "./vector_db"
    COLLECTION_NAME = "music_embeddings"
    
    # File processing settings
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    BATCH_SIZE = 8
```

## Programmatic Usage

### Basic Usage

```python
from ingestion_pipeline import MusicIngestionPipeline

# Initialize pipeline
pipeline = MusicIngestionPipeline()

# Process music files
results = pipeline.run_pipeline(music_directory="./music_files")

# Search for similar music
similar_results = pipeline.search_similar_music("query_song.mp3", n_results=10)

# Export results
pipeline.export_pipeline_results("results.json")
```

### Advanced Usage

```python
from ingestion_pipeline import MusicIngestionPipeline
from config import Config

# Customize configuration
Config.DURATION = 20  # 20-second segments
Config.BATCH_SIZE = 4  # Process 4 files at a time

# Initialize pipeline
pipeline = MusicIngestionPipeline()

# Process single file
result = pipeline.process_single_file("song.mp3")
if result:
    print(f"Processed {len(result['segments'])} segments")
    
    # Save to vector store
    pipeline.save_to_vector_store([result])
```

## API Reference

### MusicIngestionPipeline

#### Methods

- `find_music_files(directory)`: Find all supported music files in a directory
- `process_single_file(file_path)`: Process a single music file
- `run_pipeline(music_directory, batch_size)`: Run the complete pipeline
- `search_similar_music(query_file_path, n_results)`: Search for similar music
- `save_to_vector_store(processed_results)`: Save results to vector store
- `export_pipeline_results(output_file)`: Export results to JSON

### AudioProcessor

#### Methods

- `load_audio(file_path)`: Load and preprocess audio file
- `extract_features(audio_data)`: Extract audio features
- `segment_audio(audio_data, segment_duration)`: Segment audio into chunks
- `get_audio_metadata(file_path)`: Extract file metadata

### EmbeddingGenerator

#### Methods

- `generate_audio_embedding(audio_data)`: Generate embedding from raw audio
- `generate_feature_embedding(features)`: Generate embedding from audio features
- `generate_combined_embedding(audio_data, features)`: Generate combined embedding
- `batch_generate_embeddings(audio_segments, feature_segments)`: Batch process embeddings

### VectorStore

#### Methods

- `add_embeddings(embeddings, metadata, ids)`: Add embeddings to vector store
- `search_similar(query_embedding, n_results, filter_metadata)`: Search for similar embeddings
- `get_embedding_by_id(embedding_id)`: Retrieve embedding by ID
- `update_embedding(embedding_id, new_embedding, new_metadata)`: Update existing embedding
- `delete_embedding(embedding_id)`: Delete embedding
- `get_collection_stats()`: Get collection statistics
- `export_metadata(output_file)`: Export metadata to JSON

## Example Output

### Pipeline Results

```
==================================================
PIPELINE RESULTS
==================================================
Success: True
Total files found: 50
Files processed: 48
Files failed: 2
Total segments created: 240
Processing time: 125.34 seconds
Vector store embeddings: 240
Embedding dimension: 768
```

### Similar Music Search

```
Found 10 similar music pieces:
1. song1.mp3 (Segment 0)
   Tempo: 120.5 BPM
   Duration: 30.0s
   Distance: 0.1234

2. song2.mp3 (Segment 1)
   Tempo: 118.2 BPM
   Duration: 30.0s
   Distance: 0.1456
```

## Performance Considerations

- **GPU Acceleration**: The pipeline automatically uses GPU if available for faster embedding generation
- **Batch Processing**: Process multiple files in batches to optimize memory usage
- **Segment Duration**: Adjust segment duration based on your use case (longer segments = fewer embeddings but more context)
- **Vector Database**: ChromaDB provides efficient similarity search for large collections

## Troubleshooting

### Common Issues

1. **Audio file not supported**: Ensure the file format is in the supported formats list
2. **Memory issues**: Reduce batch size or segment duration
3. **Model download issues**: Check internet connection for downloading transformer models
4. **Vector store errors**: Ensure the vector_db directory has write permissions

### Logging

The pipeline uses Python's logging module. Set log level to DEBUG for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [librosa](https://librosa.org/) for audio processing
- [Transformers](https://huggingface.co/transformers/) for the Wav2Vec2 model
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [PyDub](https://github.com/jiaaro/pydub) for audio format handling 