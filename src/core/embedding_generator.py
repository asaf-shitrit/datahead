import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional
import logging
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings from audio features using transformer models"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.MODEL_NAME
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = None
        self.pca = None
        self.scaler = StandardScaler()
        
        logger.info(f"Initializing embedding generator with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model and feature extractor"""
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_audio_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Generate embedding from raw audio data using the transformer model
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Prepare input for the model
            inputs = self.feature_extractor(
                audio_data, 
                sampling_rate=Config.SAMPLE_RATE, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state as embedding
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Average over time dimension
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating audio embedding: {str(e)}")
            raise
    
    def generate_feature_embedding(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate embedding from extracted audio features
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Combined feature embedding
        """
        try:
            feature_vectors = []
            
            # Process mel spectrogram
            if 'mel_spectrogram' in features:
                mel_spec = features['mel_spectrogram']
                # Flatten and take statistics
                mel_flat = mel_spec.flatten()
                mel_stats = [
                    np.mean(mel_flat),
                    np.std(mel_flat),
                    np.min(mel_flat),
                    np.max(mel_flat),
                    np.percentile(mel_flat, 25),
                    np.percentile(mel_flat, 75)
                ]
                feature_vectors.extend(mel_stats)
            
            # Process MFCC
            if 'mfcc' in features:
                mfcc = features['mfcc']
                mfcc_stats = [
                    np.mean(mfcc),
                    np.std(mfcc),
                    np.min(mfcc),
                    np.max(mfcc)
                ]
                feature_vectors.extend(mfcc_stats)
            
            # Process spectral centroid
            if 'spectral_centroid' in features:
                spec_cent = features['spectral_centroid']
                spec_stats = [
                    np.mean(spec_cent),
                    np.std(spec_cent),
                    np.min(spec_cent),
                    np.max(spec_cent)
                ]
                feature_vectors.extend(spec_stats)
            
            # Process tempo
            if 'tempo' in features:
                feature_vectors.append(features['tempo'])
            
            # Process chroma features
            if 'chroma' in features:
                chroma = features['chroma']
                chroma_stats = [
                    np.mean(chroma),
                    np.std(chroma),
                    np.min(chroma),
                    np.max(chroma)
                ]
                feature_vectors.extend(chroma_stats)
            
            # Convert to numpy array
            feature_vector = np.array(feature_vectors, dtype=np.float32)
            
            # Normalize the feature vector
            feature_vector = self.scaler.fit_transform(feature_vector.reshape(1, -1)).flatten()
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error generating feature embedding: {str(e)}")
            raise
    
    def generate_combined_embedding(self, audio_data: np.ndarray, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate a combined embedding using both audio and feature embeddings
        
        Args:
            audio_data: Raw audio data
            features: Extracted audio features
            
        Returns:
            Combined embedding vector
        """
        try:
            # Generate audio embedding
            audio_embedding = self.generate_audio_embedding(audio_data)
            
            # Generate feature embedding
            feature_embedding = self.generate_feature_embedding(features)
            
            # Combine embeddings (you can experiment with different combination strategies)
            # For now, we'll concatenate them
            combined_embedding = np.concatenate([
                audio_embedding.flatten(),
                feature_embedding
            ])
            
            # Apply PCA if the combined embedding is too large
            if combined_embedding.shape[0] > Config.EMBEDDING_DIM:
                if self.pca is None:
                    self.pca = PCA(n_components=Config.EMBEDDING_DIM)
                    combined_embedding = self.pca.fit_transform(combined_embedding.reshape(1, -1)).flatten()
                else:
                    combined_embedding = self.pca.transform(combined_embedding.reshape(1, -1)).flatten()
            
            return combined_embedding
            
        except Exception as e:
            logger.error(f"Error generating combined embedding: {str(e)}")
            raise
    
    def batch_generate_embeddings(self, audio_segments: List[np.ndarray], 
                                 feature_segments: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of audio segments
        
        Args:
            audio_segments: List of audio data arrays
            feature_segments: List of feature dictionaries
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, (audio_segment, feature_segment) in enumerate(zip(audio_segments, feature_segments)):
            try:
                embedding = self.generate_combined_embedding(audio_segment, feature_segment)
                embeddings.append(embedding)
                logger.info(f"Generated embedding {i+1}/{len(audio_segments)}")
            except Exception as e:
                logger.error(f"Error generating embedding for segment {i}: {str(e)}")
                # Add a zero embedding as fallback
                embeddings.append(np.zeros(Config.EMBEDDING_DIM))
        
        return embeddings 