import pickle
import os
import sys
import json
import time
import sqlite3
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.audio_utils import (
    load_audio, extract_features, load_features, compute_feature_similarity,
    extract_mfcc, dtw_distance
)

# Default paths
DEFAULT_DB_PATH = os.path.join(project_root, "data", "fingerprints.db")
DEFAULT_MAPPING_PATH = os.path.join(project_root, "data", "song_mapping.json")
DEFAULT_FEATURE_DIR = os.path.join(project_root, "data", "features")

# Default confidence thresholds
FINGERPRINT_CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence for fingerprint match
DTW_SIMILARITY_THRESHOLD = 0.60          # Minimum similarity for DTW match
DEFAULT_MAX_RESULTS = 4                  # Best match + 3 alternatives

def load_song_mapping(mapping_path: str = DEFAULT_MAPPING_PATH) -> Dict[str, Any]:
    """
    Load song mapping from JSON file.
    
    Args:
        mapping_path: Path to the mapping file
        
    Returns:
        Dictionary mapping song IDs to metadata
    """
    try:
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading song mapping: {e}")
        return {}

def compute_fingerprint(audio, sr):
    """Simple placeholder for fingerprint computation"""
    # This is a simplified version - in a real app you'd use a proper fingerprinting algorithm
    import hashlib
    
    # Create a fingerprint from the audio (very simplified)
    fingerprints = []
    for i in range(0, len(audio), sr // 10):  # every 0.1 seconds
        if i + sr // 5 <= len(audio):
            chunk = audio[i:i + sr // 5]
            # Create a hash of this chunk
            hash_obj = hashlib.md5(chunk.tobytes())
            hash_value = hash_obj.hexdigest()
            fingerprints.append((hash_value, i))
    
    return fingerprints

def recognize_fingerprint(db_conn: sqlite3.Connection, filepath: str) -> Dict[str, Any]:
    """
    Recognize song using audio fingerprinting.
    
    Args:
        db_conn: Database connection
        filepath: Path to audio file to recognize
        
    Returns:
        Dictionary with recognition results
    """
    # Measure time
    start_time = time.time()
    
    # Recognize the file
    try:
        # Load audio
        audio, sr = load_audio(filepath)
        
        # Compute fingerprints from the audio
        fingerprints = compute_fingerprint(audio, sr)
        
        # Create a dict to count matches for each song
        song_matches = {}
        
        # Query each fingerprint hash
        cursor = db_conn.cursor()
        for hash_value, offset in fingerprints:
            # Look for matching hashes in the database
            cursor.execute(
                "SELECT song_id, offset FROM fingerprints WHERE hash = ?",
                (hash_value,)
            )
            
            matches = cursor.fetchall()
            for song_id, db_offset in matches:
                # Calculate time offset
                delta = db_offset - offset
                
                # Count matches per song with same offset
                if song_id not in song_matches:
                    song_matches[song_id] = {}
                
                if delta not in song_matches[song_id]:
                    song_matches[song_id][delta] = 0
                
                song_matches[song_id][delta] += 1
        
        # Calculate confidence for each song
        results = []
        for song_id, offsets in song_matches.items():
            # Get the offset with the most matches
            max_offset = max(offsets.items(), key=lambda x: x[1])
            max_matches = max_offset[1]
            
            # Get song details
            cursor.execute(
                "SELECT title, artist FROM songs WHERE song_id = ?",
                (song_id,)
            )
            song_info = cursor.fetchone()
            
            if song_info:
                title, artist = song_info
                
                # Calculate confidence (normalized by fingerprint count)
                confidence = max_matches / len(fingerprints) if fingerprints else 0
                
                results.append({
                    'song_id': song_id,
                    'song_name': title,
                    'artist': artist,
                    'confidence': confidence,
                    'matches': max_matches,
                    'offset': max_offset[0]
                })
        
        # Sort results by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        return {
            'success': len(results) > 0,
            'matches': results,
            'elapsed_time': elapsed_time
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }

def recognize_dtw(query_features: Dict[str, np.ndarray], 
                 song_mapping: Dict[str, Any],
                 feature_dir: str = DEFAULT_FEATURE_DIR,
                 max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
    """
    Recognize song using DTW-based similarity search on audio features.
    
    Args:
        query_features: Features extracted from query audio
        song_mapping: Dictionary mapping song IDs to metadata
        feature_dir: Directory containing feature files
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with recognition results
    """
    # Measure time
    start_time = time.time()
    
    # Calculate similarity for each song in the database
    similarities = []
    
    for song_id, song_info in song_mapping.items():
        # Load song features
        song_file = os.path.join(feature_dir, f"{song_id}.npy")
        if not os.path.exists(song_file):
            continue
            
        try:
            song_features = load_features(song_file)
            
            # Calculate similarity using DTW
            similarity = compute_feature_similarity(query_features, song_features)
            
            # Add to results
            similarities.append({
                'song_id': song_id,
                'artist': song_info.get('artist', 'Unknown'),
                'title': song_info.get('title', 'Unknown'),
                'similarity': similarity
            })
        except Exception as e:
            print(f"Error processing song {song_id}: {e}")
            continue
    
    # Sort by similarity (higher is better)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Take top matches
    top_matches = similarities[:max_results]
    
    return {
        'success': len(top_matches) > 0,
        'matches': top_matches,
        'elapsed_time': time.time() - start_time
    }

def combine_results(fingerprint_results: Dict[str, Any], 
                   dtw_results: Dict[str, Any],
                   song_mapping: Dict[str, Any],
                   fingerprint_threshold: float = FINGERPRINT_CONFIDENCE_THRESHOLD,
                   dtw_threshold: float = DTW_SIMILARITY_THRESHOLD,
                   max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
    """
    Combine results from fingerprint and DTW recognition.
    
    Args:
        fingerprint_results: Results from fingerprint recognition
        dtw_results: Results from DTW recognition
        song_mapping: Dictionary mapping song IDs to metadata
        fingerprint_threshold: Minimum confidence for fingerprint match
        dtw_threshold: Minimum similarity for DTW match
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with combined results
    """
    # Track all matches with their scores
    all_matches = {}
    
    # Process fingerprint matches
    if fingerprint_results.get('success', False) and fingerprint_results.get('matches'):
        for match in fingerprint_results['matches']:
            song_id = match['song_id']
            confidence = match['confidence']
            
            # Only consider matches above threshold
            if confidence >= fingerprint_threshold:
                all_matches[song_id] = {
                    'song_id': song_id,
                    'artist': match.get('artist', 'Unknown'),
                    'title': match.get('song_name', 'Unknown'),
                    'fingerprint_confidence': confidence,
                    'dtw_similarity': 0,
                    'combined_score': confidence * 0.7,  # 70% weight for fingerprint
                    'match_type': 'fingerprint'
                }
    
    # Process DTW matches
    if dtw_results.get('success', False) and dtw_results.get('matches'):
        for match in dtw_results['matches']:
            song_id = match['song_id']
            similarity = match['similarity']
            
            # Only consider matches above threshold
            if similarity >= dtw_threshold:
                if song_id in all_matches:
                    # Update existing match
                    all_matches[song_id]['dtw_similarity'] = similarity
                    # Combine scores (70% fingerprint, 30% DTW)
                    fp_score = all_matches[song_id]['fingerprint_confidence']
                    all_matches[song_id]['combined_score'] = (fp_score * 0.7) + (similarity * 0.3)
                    # If it matched both ways, it's likely a stronger match
                    all_matches[song_id]['match_type'] = 'combined'
                else:
                    # Add new match
                    all_matches[song_id] = {
                        'song_id': song_id,
                        'artist': match['artist'],
                        'title': match['title'],
                        'fingerprint_confidence': 0,
                        'dtw_similarity': similarity,
                        'combined_score': similarity * 0.3,  # Lower weight for DTW-only
                        'match_type': 'dtw'
                    }
    
    # Convert to list and sort by combined score
    combined_matches = list(all_matches.values())
    combined_matches.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Limit to max results
    top_matches = combined_matches[:max_results]
    
    # Organize results with best match and alternatives
    if top_matches:
        best_match = top_matches[0]
        alternatives = top_matches[1:] if len(top_matches) > 1 else []
        
        return {
            'success': True,
            'best_match': best_match,
            'alternatives': alternatives,
            'elapsed_time': max(
                fingerprint_results.get('elapsed_time', 0),
                dtw_results.get('elapsed_time', 0)
            )
        }
    else:
        return {
            'success': False,
            'error': 'No matches found',
            'elapsed_time': max(
                fingerprint_results.get('elapsed_time', 0),
                dtw_results.get('elapsed_time', 0)
            )
        }

def recognize_song(filepath: str, 
                  db_path: str = DEFAULT_DB_PATH,
                  mapping_path: str = DEFAULT_MAPPING_PATH,
                  feature_dir: str = DEFAULT_FEATURE_DIR,
                  fingerprint_threshold: float = FINGERPRINT_CONFIDENCE_THRESHOLD,
                  dtw_threshold: float = DTW_SIMILARITY_THRESHOLD,
                  max_results: int = DEFAULT_MAX_RESULTS) -> Dict[str, Any]:
    """
    Recognize a song using the two-tiered approach.
    
    Args:
        filepath: Path to the audio file to recognize
        db_path: Path to the fingerprint database
        mapping_path: Path to the song mapping file
        feature_dir: Directory containing feature files
        fingerprint_threshold: Minimum confidence for fingerprint match
        dtw_threshold: Minimum similarity for DTW match
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with recognition results
    """
    # Load song mapping
    song_mapping = load_song_mapping(mapping_path)
    if not song_mapping:
        return {
            'success': False,
            'error': f'No songs found in mapping file: {mapping_path}'
        }
    
    # Connect to database
    db_conn = sqlite3.connect(db_path)
    
    # Step 1: Try fingerprint recognition
    print("Running fingerprint recognition...")
    fingerprint_results = recognize_fingerprint(db_conn, filepath)
    
    has_confident_match = False
    
    if fingerprint_results.get('success', False) and fingerprint_results.get('matches'):
        for match in fingerprint_results['matches']:
            confidence = match['confidence']
            if confidence > 1:
                confidence = confidence / 100
            
            if confidence >= fingerprint_threshold:
                has_confident_match = True
                break
    
    # Step 2: Try DTW recognition if needed
    dtw_results = {'success': False, 'matches': []}
    if not has_confident_match or len(fingerprint_results.get('matches', [])) == 0:
        print("No confident fingerprint match found. Running DTW recognition...")
        
        # Extract features for DTW recognition
        audio, sr = load_audio(filepath)
        query_features = extract_features(audio, sr)
        
        # Run DTW recognition
        dtw_results = recognize_dtw(query_features, song_mapping, feature_dir, max_results)
    
    # Combine results
    results = combine_results(
        fingerprint_results, dtw_results, song_mapping,
        fingerprint_threshold, dtw_threshold, max_results
    )
    
    # Close the database connection
    db_conn.close()
    
    return results

def format_results(results: Dict[str, Any]) -> str:
    """
    Format recognition results as human-readable text.
    
    Args:
        results: Dictionary with recognition results
        
    Returns:
        String with formatted results
    """
    if not results.get('success', False):
        return f"Recognition failed: {results.get('error', 'Unknown error')}"
    
    output = []
    output.append(f"Recognition completed in {results.get('elapsed_time', 0):.2f} seconds")
    
    best_match = results.get('best_match')
    if best_match:
        output.append("\n=== Best Match ===")
        output.append(f"Song: {best_match['artist']} - {best_match['title']}")
        output.append(f"Match type: {best_match['match_type']}")
        output.append(f"Fingerprint confidence: {best_match['fingerprint_confidence']*100:.1f}%")
        output.append(f"DTW similarity: {best_match['dtw_similarity']*100:.1f}%")
        output.append(f"Combined score: {best_match['combined_score']*100:.1f}%")
    
    alternatives = results.get('alternatives', [])
    if alternatives:
        output.append("\n=== Alternative Matches ===")
        for i, alt in enumerate(alternatives):
            output.append(f"\n#{i+1}: {alt['artist']} - {alt['title']}")
            output.append(f"Match type: {alt['match_type']}")
            output.append(f"Fingerprint confidence: {alt['fingerprint_confidence']*100:.1f}%")
            output.append(f"DTW similarity: {alt['dtw_similarity']*100:.1f}%")
            output.append(f"Combined score: {alt['combined_score']*100:.1f}%")
    
    return "\n".join(output)


if __name__ == "__main__":
    # Simple command-line testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Recognize songs from audio input.")
    parser.add_argument("input", help="Path to audio file to recognize")
    args = parser.parse_args()
    
    results = recognize_song(args.input)
    print(format_results(results))
