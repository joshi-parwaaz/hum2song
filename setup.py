#!/usr/bin/env python
"""
setup.py - Setup script for the hum2song project.

This script helps users set up the project with their own dataset.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/songs_mp3",
        "data/features", 
        "data/hums",
        "scripts/__pycache__",
        "utils/__pycache__"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_empty_files():
    """Create empty files if they don't exist."""
    files = {
        "data/song_mapping.json": "{}",
        "scripts/__init__.py": "",
        "utils/__init__.py": ""
    }
    
    for filepath, content in files.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✓ Created file: {filepath}")
        else:
            print(f"✓ File exists: {filepath}")

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "numpy", "scipy", "librosa", "streamlit", 
        "tqdm", "sounddevice", "matplotlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def validate_audio_files(songs_dir="data/songs_mp3"):
    """Check if there are audio files in the songs directory."""
    if not os.path.exists(songs_dir):
        print(f"⚠️  Songs directory '{songs_dir}' does not exist")
        return False
    
    audio_extensions = ['.mp3', '.wav', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(songs_dir).glob(f"**/*{ext}"))
    
    if not audio_files:
        print(f"⚠️  No audio files found in '{songs_dir}'")
        print("Please add some MP3/WAV files to get started")
        return False
    
    print(f"✓ Found {len(audio_files)} audio files in '{songs_dir}'")
    return True

def run_ingest(songs_dir="data/songs_mp3"):
    """Run the ingest script to process audio files."""
    if not validate_audio_files(songs_dir):
        return False
    
    print("\n🎵 Processing audio files...")
    try:
        # Import and run ingest
        from scripts.ingest import ingest_folder
        ingest_folder(songs_dir)
        print("✓ Audio files processed successfully!")
        return True
    except Exception as e:
        print(f"✗ Error processing audio files: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup hum2song project")
    parser.add_argument("--songs-dir", default="data/songs_mp3", 
                       help="Directory containing audio files")
    parser.add_argument("--skip-ingest", action="store_true",
                       help="Skip audio processing step")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies and setup")
    
    args = parser.parse_args()
    
    print("🎵 Hum2Song Setup")
    print("=" * 50)
    
    # Step 1: Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Step 2: Create empty files
    print("\n2. Creating initial files...")
    create_empty_files()
    
    # Step 3: Check dependencies
    print("\n3. Checking dependencies...")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Setup incomplete - missing dependencies")
        print("Please install missing packages and run setup again")
        return 1
    
    if args.check_only:
        print("\n✅ Setup check complete!")
        return 0
    
    # Step 4: Process audio files
    if not args.skip_ingest:
        print(f"\n4. Processing audio files from '{args.songs_dir}'...")
        if validate_audio_files(args.songs_dir):
            if run_ingest(args.songs_dir):
                print("\n✅ Setup complete!")
                print("\nYou can now run the app with: streamlit run app.py")
            else:
                print("\n⚠️  Setup complete but audio processing failed")
                print("You can manually run: python scripts/ingest.py data/songs_mp3")
        else:
            print(f"\n⚠️  No audio files found in '{args.songs_dir}'")
            print("Please add audio files and run: python scripts/ingest.py data/songs_mp3")
    else:
        print("\n✅ Setup complete (skipped audio processing)!")
        print("Don't forget to run: python scripts/ingest.py data/songs_mp3")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
