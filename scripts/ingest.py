#!/usr/bin/env python
"""
ingest.py - Ingest audio files into the system.

This script processes audio files to extract fingerprints and features
for use in song recognition.
"""

import os
import sys
import json
import time
import glob
import hashlib
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.audio_utils import (
    load_audio, extract_features, save_features, extract_mfcc
)

# Default paths
DEFAULT_DB_PATH = os.path.join(project_root, "data", "fingerprints.db")
DEFAULT_MAPPING_PATH = os.path.join(project_root, "data", "song_mapping.json")
DEFAULT_FEATURE_DIR = os.path.join(project_root, "data", "features")

# Constants for fingerprinting
FINGERPRINT_WINDOW_SIZE = 4096
FINGERPRINT_HOP_SIZE = 512
FINGERPRINT_FAN_VALUE = 15
FINGERPRINT_MIN_TIME_DELTA = 1
FINGERPRINT_MAX_TIME_DELTA = 10

def compute_fingerprint(audio: np.ndarray, sample_rate: int) -> List[Tuple[str, int]]:
    """
    Compute audio fingerprints for a song.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        
    Returns:
        List of (hash, offset) tuples
    """
    # Simple fingerprinting method (for demonstration)
    fingerprints = []
    for i in range(0, len(audio), sample_rate // 10):  # every 0.1 seconds
        if i + sample_rate // 5 <= len(audio):
            chunk = audio[i:i + sample_rate // 5]
            # Create a hash of this chunk
            hash_obj = hashlib.md5(chunk.tobytes())
            hash_value = hash_obj.hexdigest()
            fingerprints.append((hash_value, i))
    
    return fingerprints

def create_fingerprint_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Create a fingerprint database.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Database connection
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS songs (
        song_id TEXT PRIMARY KEY,
        title TEXT,
        artist TEXT,
        duration REAL,
        file_path TEXT,
        fingerprinted INTEGER DEFAULT 0
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fingerprints (
        hash TEXT,
        song_id TEXT,
        offset INTEGER,
        FOREIGN KEY (song_id) REFERENCES songs(song_id)
    )
    ''')
    
    # Create index for fast lookup
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON fingerprints (hash)')
    
    # Commit changes
    conn.commit()
    
    return conn

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

def save_song_mapping(song_mapping: Dict[str, Any], mapping_path: str = DEFAULT_MAPPING_PATH) -> None:
    """
    Save song mapping to JSON file.
    
    Args:
        song_mapping: Dictionary mapping song IDs to metadata
        mapping_path: Path to the mapping file
    """
    try:
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump(song_mapping, f, indent=2)
    except Exception as e:
        print(f"Error saving song mapping: {e}")

def process_song(filepath: str, feature_dir: str = DEFAULT_FEATURE_DIR) -> Dict[str, Any]:
    """
    Process a song file to extract features and metadata.
    
    Args:
        filepath: Path to song file
        feature_dir: Directory to save features
        
    Returns:
        Dictionary with song metadata
    """
    # Extract filename and path info
    file = Path(filepath)
    filename = file.name
    
    # Generate a unique song ID
    song_id = hashlib.md5(filename.encode()).hexdigest()
    
    # Get metadata from filename (very basic)
    parts = file.stem.split(" - ", 1)
    if len(parts) == 2:
        artist, title = parts
    else:
        artist = "Unknown"
        title = file.stem
    
    # Process audio
    try:
        # Load audio
        audio, sr = load_audio(filepath)
        
        # Calculate duration
        duration = len(audio) / sr
        
        # Extract and save features
        features = extract_features(audio, sr)
        feature_path = os.path.join(feature_dir, f"{song_id}.npy")
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        save_features(features, feature_path)
        
        # Compute fingerprints
        fingerprints = compute_fingerprint(audio, sr)
        
        return {
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "duration": duration,
            "file_path": str(file.absolute()),
            "fingerprints": fingerprints,
            "success": True
        }
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "file_path": str(file.absolute()),
            "success": False,
            "error": str(e)
        }

def ingest_folder(folder_path: str, 
                 db_path: str = DEFAULT_DB_PATH,
                 mapping_path: str = DEFAULT_MAPPING_PATH,
                 feature_dir: str = DEFAULT_FEATURE_DIR,
                 max_workers: int = None,
                 extensions: List[str] = [".mp3", ".wav", ".flac", ".ogg"]) -> None:
    """
    Ingest all audio files from a folder.
    
    Args:
        folder_path: Path to folder containing audio files
        db_path: Path to fingerprint database
        mapping_path: Path to song mapping file
        feature_dir: Directory to save features
        max_workers: Maximum number of worker processes
        extensions: List of file extensions to process
    """
    # Find all audio files
    audio_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, f"**/*{ext}")
        audio_files.extend(glob.glob(pattern, recursive=True))
    
    if not audio_files:
        print(f"No audio files found in {folder_path}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create database
    conn = create_fingerprint_db(db_path)
    cursor = conn.cursor()
    
    # Load existing song mapping
    song_mapping = load_song_mapping(mapping_path)
    
    # Process songs in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_song, file, feature_dir): file
            for file in audio_files
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="Processing songs"):
            file = future_to_file[future]
            try:
                result = future.result()
                
                if not result.get("success", False):
                    print(f"Failed to process {file}: {result.get('error')}")
                    continue
                
                # Add to database
                song_id = result["song_id"]
                
                # Check if song already exists in database
                cursor.execute("SELECT 1 FROM songs WHERE song_id = ?", (song_id,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing song
                    cursor.execute("""
                    UPDATE songs SET 
                    title = ?, artist = ?, duration = ?, file_path = ?, fingerprinted = 0
                    WHERE song_id = ?
                    """, (
                        result["title"],
                        result["artist"],
                        result.get("duration", 0),
                        result["file_path"],
                        song_id
                    ))
                    
                    # Delete existing fingerprints
                    cursor.execute("DELETE FROM fingerprints WHERE song_id = ?", (song_id,))
                else:
                    # Insert new song
                    cursor.execute("""
                    INSERT INTO songs (song_id, title, artist, duration, file_path, fingerprinted)
                    VALUES (?, ?, ?, ?, ?, 0)
                    """, (
                        song_id,
                        result["title"],
                        result["artist"],
                        result.get("duration", 0),
                        result["file_path"]
                    ))
                
                # Insert fingerprints
                fingerprints = result.get("fingerprints", [])
                if fingerprints:
                    cursor.executemany("""
                    INSERT INTO fingerprints (hash, song_id, offset)
                    VALUES (?, ?, ?)
                    """, [(fp[0], song_id, fp[1]) for fp in fingerprints])
                    
                    # Mark as fingerprinted
                    cursor.execute("UPDATE songs SET fingerprinted = 1 WHERE song_id = ?", (song_id,))
                
                # Add to song mapping
                song_mapping[song_id] = {
                    "title": result["title"],
                    "artist": result["artist"],
                    "duration": result.get("duration", 0),
                    "file_path": result["file_path"]
                }
                
                # Commit changes
                conn.commit()
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Save song mapping
    save_song_mapping(song_mapping, mapping_path)
    
    # Close database connection
    conn.close()
    
    # Print summary
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM songs")
    song_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM fingerprints")
    fingerprint_count = cursor.fetchone()[0]
    
    print(f"Ingestion complete:")
    print(f"- {song_count} songs in database")
    print(f"- {fingerprint_count} fingerprints in database")
    print(f"- {len(song_mapping)} songs in mapping file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest audio files into the system.")
    parser.add_argument("folder", help="Path to folder containing audio files")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to fingerprint database")
    parser.add_argument("--mapping", default=DEFAULT_MAPPING_PATH, help="Path to song mapping file")
    parser.add_argument("--features", default=DEFAULT_FEATURE_DIR, help="Directory to save features")
    parser.add_argument("--workers", type=int, default=None, help="Maximum number of worker processes")
    parser.add_argument("--extensions", nargs="+", default=[".mp3", ".wav", ".flac", ".ogg"], 
                      help="File extensions to process")
    
    args = parser.parse_args()
    
    ingest_folder(
        args.folder,
        db_path=args.db,
        mapping_path=args.mapping,
        feature_dir=args.features,
        max_workers=args.workers,
        extensions=args.extensions
    )
