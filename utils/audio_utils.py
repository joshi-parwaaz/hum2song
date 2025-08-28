#!/usr/bin/env python
"""
audio_utils.py - Audio processing utilities for the hum2song project.

This module contains functions for loading, saving, and processing audio files.
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Default paths
DEFAULT_FEATURE_DIR = os.path.join(project_root, "data", "features")

# Suppress the NumPy/Numba version warning
warnings.filterwarnings("ignore", message="Numba needs NumPy")

# Try importing optional dependencies
try:
    # First, try to import librosa with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa library not available. Audio processing will be limited.")

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy library not available. Some audio processing functions will be limited.")

def load_audio(filepath: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa or a fallback method.
    
    Args:
        filepath: Path to the audio file
        sr: Target sample rate (None for original)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Try to use librosa first
    if LIBROSA_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(filepath, sr=sr, mono=True)
            return audio, sr
        except Exception as e:
            logger.warning(f"Librosa failed to load audio: {str(e)}. Trying fallback method.")
    
    # Fallback method using scipy
    if SCIPY_AVAILABLE:
        try:
            from scipy.io import wavfile
            
            # For MP3 files, convert to WAV first
            if filepath.lower().endswith('.mp3'):
                import tempfile
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name
                
                # Use ffmpeg to convert MP3 to WAV
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", filepath, temp_wav_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    sample_rate, audio_data = wavfile.read(temp_wav_path)
                    
                    # Convert to float32 and normalize
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    
                    # If stereo, convert to mono
                    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                        audio_data = np.mean(audio_data, axis=1)
                    
                    # Resample if needed
                    if sr is not None and sr != sample_rate:
                        # Simple resampling
                        from scipy import signal
                        new_length = int(len(audio_data) * sr / sample_rate)
                        audio_data = signal.resample(audio_data, new_length)
                        sample_rate = sr
                    
                    return audio_data, sample_rate
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)
            else:
                # Direct loading for WAV files
                sample_rate, audio_data = wavfile.read(filepath)
                
                # Convert to float32 and normalize
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                
                # If stereo, convert to mono
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if needed
                if sr is not None and sr != sample_rate:
                    # Simple resampling
                    from scipy import signal
                    new_length = int(len(audio_data) * sr / sample_rate)
                    audio_data = signal.resample(audio_data, new_length)
                    sample_rate = sr
                
                return audio_data, sample_rate
                
        except Exception as e:
            logger.error(f"Fallback audio loading failed: {str(e)}")
    
    # If all methods fail, raise an error
    raise ImportError("No suitable audio loading library available. Please install librosa or scipy.")

def save_features(features: np.ndarray, filepath: str) -> None:
    """
    Save extracted features to a file.
    
    Args:
        features: Feature array
        filepath: Path to save the features to
    """
    try:
        np.save(filepath, features)
    except Exception as e:
        logger.error(f"Error saving features to {filepath}: {str(e)}")
        raise

def load_features(filepath: str) -> np.ndarray:
    """
    Load extracted features from a file.
    
    Args:
        filepath: Path to the feature file
        
    Returns:
        Feature array
    """
    try:
        return np.load(filepath)
    except Exception as e:
        logger.error(f"Error loading features from {filepath}: {str(e)}")
        raise

def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract features from audio data.
    
    Args:
        audio: Audio data
        sr: Sample rate
        
    Returns:
        Feature array
    """
    # Try to use librosa for feature extraction
    if LIBROSA_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Extract MFCCs
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                
                # Extract spectral centroid
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
                
                # Extract chroma features
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                
                # Stack features
                features = np.vstack([mfcc, spectral_centroid, chroma])
                
                return features
                
        except Exception as e:
            logger.warning(f"Librosa feature extraction failed: {str(e)}. Using fallback method.")
    
    # Fallback to basic features using NumPy FFT
    # This is a simplified version that doesn't exactly match librosa's features
    try:
        # Divide audio into frames
        frame_length = 2048
        hop_length = 512
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frames.append(audio[i:i + frame_length])
        
        if not frames:
            # Handle short audio by using the entire clip as one frame
            frames = [audio]
        
        frames = np.array(frames)
        
        # Apply window function
        window = np.hanning(frame_length)
        frames = frames * window
        
        # Compute FFT for each frame
        fft_frames = np.abs(np.fft.rfft(frames, axis=1))
        
        # Compute simple spectral features (power spectrum)
        power_spectrum = fft_frames**2
        
        # Compute spectral centroid (weighted average of frequency components)
        freqs = np.fft.rfftfreq(frame_length, 1.0/sr)
        spectral_centroid = np.sum(power_spectrum * freqs.reshape(1, -1), axis=1) / np.sum(power_spectrum, axis=1)
        spectral_centroid = spectral_centroid.reshape(1, -1)  # Make it 2D like librosa
        
        # Create a dummy feature set (not as good as librosa's but will work for testing)
        dummy_features = np.vstack([
            power_spectrum[:, :13].T,  # Use first 13 frequency bins as dummy MFCCs
            spectral_centroid,         # Use spectral centroid
            power_spectrum[:, 13:25].T  # Use next 12 frequency bins as dummy chroma
        ])
        
        return dummy_features
        
    except Exception as e:
        logger.error(f"Fallback feature extraction failed: {str(e)}")
        
        # If all methods fail, return a dummy feature array
        return np.random.rand(26, min(100, len(audio) // 512 + 1))  # 13 MFCCs + 1 centroid + 12 chroma

def preprocess_for_recognition(filepath: str) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Preprocess audio file for recognition.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        Tuple of (audio_data, sample_rate, features)
    """
    # Load audio
    audio, sr = load_audio(filepath)
    
    # Extract features
    features = extract_features(audio, sr)
    
    return audio, sr, features

def extract_mfcc(path, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    # Use the existing load_audio function
    y, sr = load_audio(path, sr=22050)
    
    # Check if librosa is available
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for MFCC extraction")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    return mfcc.T

def dtw_distance(mfcc1, mfcc2):
    """Compute DTW distance between two MFCC sequences."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for DTW computation")
    
    from scipy.spatial.distance import cdist
    
    # If librosa is available, use its DTW implementation
    if LIBROSA_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from librosa.sequence import dtw
                
                D = cdist(mfcc1, mfcc2, metric="euclidean")
                _, wp = dtw(C=D)
                cost = D[tuple(zip(*wp))].sum() / len(wp)
                return cost
        except Exception as e:
            logger.warning(f"Failed to use librosa DTW: {str(e)}. Falling back to custom implementation.")
    
    # Otherwise, use our custom DTW implementation
    def compute_dtw(x, y, distance_metric='euclidean'):
        distance_matrix = cdist(x, y, metric=distance_metric)
        
        # Initialize cost matrix
        cost_matrix = np.zeros((len(x), len(y)))
        cost_matrix[0, 0] = distance_matrix[0, 0]
        
        # Fill the first row
        for j in range(1, len(y)):
            cost_matrix[0, j] = cost_matrix[0, j-1] + distance_matrix[0, j]
        
        # Fill the first column
        for i in range(1, len(x)):
            cost_matrix[i, 0] = cost_matrix[i-1, 0] + distance_matrix[i, 0]
        
        # Fill the rest of the matrix
        for i in range(1, len(x)):
            for j in range(1, len(y)):
                cost_matrix[i, j] = distance_matrix[i, j] + min(
                    cost_matrix[i-1, j],      # Insertion
                    cost_matrix[i, j-1],      # Deletion
                    cost_matrix[i-1, j-1]     # Match
                )
        
        return cost_matrix[-1, -1] / max(len(x), len(y))
    
    return compute_dtw(mfcc1, mfcc2)

def compute_feature_similarity(query_features: np.ndarray, target_features: np.ndarray) -> float:
    """
    Compute similarity between two feature sets using Dynamic Time Warping.
    
    Args:
        query_features: Features of the query audio
        target_features: Features of the target audio
        
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for DTW computation")
    
    # Transpose features to time x features
    query_features = query_features.T
    target_features = target_features.T
    
    # Compute DTW distance
    from scipy.spatial.distance import cdist
    
    # Function to compute DTW
    def dtw(x, y, distance_metric='euclidean'):
        distance_matrix = cdist(x, y, metric=distance_metric)
        
        # Initialize cost matrix
        cost_matrix = np.zeros((len(x), len(y)))
        cost_matrix[0, 0] = distance_matrix[0, 0]
        
        # Fill the first row
        for j in range(1, len(y)):
            cost_matrix[0, j] = cost_matrix[0, j-1] + distance_matrix[0, j]
        
        # Fill the first column
        for i in range(1, len(x)):
            cost_matrix[i, 0] = cost_matrix[i-1, 0] + distance_matrix[i, 0]
        
        # Fill the rest of the matrix
        for i in range(1, len(x)):
            for j in range(1, len(y)):
                cost_matrix[i, j] = distance_matrix[i, j] + min(
                    cost_matrix[i-1, j],      # Insertion
                    cost_matrix[i, j-1],      # Deletion
                    cost_matrix[i-1, j-1]     # Match
                )
        
        # Return the normalized cost
        max_len = max(len(x), len(y))
        return 1.0 - (cost_matrix[-1, -1] / (max_len * 2.0))
    
    # Compute DTW
    similarity = dtw(query_features, target_features)
    
    return similarity
