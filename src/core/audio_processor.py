import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
import os
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio file processing and feature extraction"""
    
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.duration = Config.DURATION
        self.hop_length = Config.HOP_LENGTH
        self.n_mels = Config.N_MELS
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono if necessary
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Handle different audio formats
            if file_path.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(file_path)
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(self.sample_rate)
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
            else:
                audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            logger.info(f"Loaded audio file: {file_path}")
            return audio_data, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            raise
    
    def extract_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract various audio features from the audio data
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        try:
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spectrogram'] = mel_spec_db
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13,
                hop_length=self.hop_length
            )
            features['mfcc'] = mfcc
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            features['spectral_centroid'] = spectral_centroids
            
            # Extract tempo and beat features
            tempo, beats = librosa.beat.beat_track(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            features['tempo'] = tempo
            features['beats'] = beats
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            features['chroma'] = chroma
            
            # Extract harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio_data)
            features['harmonic'] = harmonic
            features['percussive'] = percussive
            
            logger.info(f"Extracted features: {list(features.keys())}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def segment_audio(self, audio_data: np.ndarray, segment_duration: int = None) -> list:
        """
        Segment audio into chunks for processing
        
        Args:
            audio_data: Audio data as numpy array
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of audio segments
        """
        if segment_duration is None:
            segment_duration = self.duration
        
        segment_length = int(segment_duration * self.sample_rate)
        segments = []
        
        for i in range(0, len(audio_data), segment_length):
            segment = audio_data[i:i + segment_length]
            if len(segment) >= segment_length // 2:  # Only include segments with at least half the target length
                segments.append(segment)
        
        logger.info(f"Created {len(segments)} segments from audio")
        return segments
    
    def get_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from audio file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing audio metadata
        """
        try:
            # Get basic file info
            file_path = Path(file_path)
            metadata = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix.lower()
            }
            
            # Get audio duration
            audio_data, _ = self.load_audio(str(file_path))
            duration = len(audio_data) / self.sample_rate
            metadata['duration'] = duration
            
            # Get additional audio properties
            features = self.extract_features(audio_data)
            metadata['tempo'] = features['tempo']
            metadata['num_beats'] = len(features['beats'])
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {'filename': file_path.name, 'error': str(e)} 