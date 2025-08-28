# Hum2Song

A song recognition system that can identify songs from both exact audio clips and hummed/sung variations. This project implements a two-tiered approach to song recognition:

1. **Fingerprinting-based matching**: Fast and precise for exact audio clips (like Shazam)
2. **DTW-based matching**: More flexible for hummed/sung variations

## Features

- **🎵 Upload & Recognize**: Upload audio files and get them identified
- **🎤 Record & Recognize**: Use your microphone to hum a song and get it identified
- **🔍 Two-tier Recognition**: Fast fingerprinting for exact matches, DTW for hummed melodies
- **🌐 Web Interface**: User-friendly Streamlit web app
- **📊 Song Database Management**: Add songs to your personal database

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd hum2song

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Songs

Place your MP3/WAV files in the `data/songs_mp3/` directory:

```
hum2song/
├── data/
│   ├── songs_mp3/           # 👈 Put your songs here
│   │   ├── Artist - Song1.mp3
│   │   ├── Artist - Song2.mp3
│   │   └── ...
│   ├── features/            # Auto-generated
│   └── fingerprints.db      # Auto-generated
```

### 3. Setup the Database

```bash
# Option 1: Use the setup script (recommended)
python setup.py

# Option 2: Manual setup
python scripts/ingest.py data/songs_mp3
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

### Two-Tiered Recognition System

1. **Audio Fingerprinting** (Primary):
   - Creates unique fingerprints for each song
   - Fast exact matching for identical audio
   - Works best with clear, studio recordings

2. **Dynamic Time Warping** (Secondary):
   - Compares MFCC (Mel-frequency cepstral coefficients) features
   - Handles tempo variations and humming
   - More flexible but computationally intensive

### Recognition Process

```
Audio Input → Feature Extraction → Fingerprint Match (if confident)
                                    ↓ (if not confident)
                                DTW Match → Combined Results
```

## Project Structure

```
hum2song/
├── app.py                   # Main Streamlit application
├── setup.py                 # Setup script for easy installation
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Data directory
│   ├── songs_mp3/          # Your song files go here
│   ├── features/           # Extracted features (auto-generated)
│   ├── hums/              # Recorded hums (auto-generated)
│   ├── fingerprints.db    # Fingerprint database (auto-generated)
│   └── song_mapping.json  # Song metadata (auto-generated)
├── scripts/               # Core processing scripts
│   ├── __init__.py
│   ├── download_songs.py  # Download sample songs
│   ├── ingest.py          # Process songs into database
│   └── recognize.py       # Song recognition logic
└── utils/                 # Utility functions
    ├── __init__.py
    └── audio_utils.py     # Audio processing utilities
```

## Usage Guide

### Adding Songs to Database

1. **Place audio files** in `data/songs_mp3/`
2. **Name files descriptively**: `Artist - Song Title.mp3`
3. **Run ingest script**: `python scripts/ingest.py data/songs_mp3`

### Using the Web App

1. **Upload audio file** or **select from samples**
2. **Adjust recognition settings** (thresholds)
3. **Click "Recognize Song"**
4. **View results** with confidence scores

### Recognition Settings

- **Fingerprint Threshold** (0.0-1.0): Lower = more matches, less accuracy
- **DTW Threshold** (0.0-1.0): Lower = more humming matches, less accuracy
- **Max Results**: Number of alternative matches to show

## Command Line Usage

### Download Sample Songs

```bash
python scripts/download_songs.py --max 10
```

### Process Songs

```bash
python scripts/ingest.py data/songs_mp3
```

### Recognize a File

```bash
python scripts/recognize.py path/to/audio/file.wav
```

## Troubleshooting

### Common Issues

1. **No songs in database**:
   - Ensure files are in `data/songs_mp3/`
   - Run `python scripts/ingest.py data/songs_mp3`

2. **Audio loading errors**:
   - Install `ffmpeg`: `pip install ffmpeg-python`
   - Check file formats (MP3, WAV supported)

3. **Recognition not working**:
   - Lower threshold values for more matches
   - Ensure songs are properly ingested
   - Check debug information in app

4. **Missing dependencies**:
   - Run `python setup.py --check-only`
   - Install missing packages: `pip install -r requirements.txt`

### Performance Tips

- **For better fingerprint matching**: Use high-quality audio files
- **For better humming recognition**: Hum clearly and match the melody closely
- **For faster processing**: Use smaller song database for testing

## Technical Details

### Dependencies

- **Core**: `numpy`, `scipy`, `librosa`
- **Web App**: `streamlit`, `streamlit-webrtc`
- **Audio Processing**: `sounddevice`, `ffmpeg-python`
- **Utilities**: `tqdm`, `matplotlib`

### Audio Processing

- **Sample Rate**: 22050 Hz (default)
- **Features**: MFCC (13 coefficients), Spectral Centroid, Chroma
- **Fingerprinting**: MD5 hash of audio chunks
- **DTW**: Custom implementation with Euclidean distance

### Database

- **SQLite**: Stores fingerprints and song metadata
- **JSON**: Song mapping and metadata
- **NumPy**: Feature vectors stored as `.npy` files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Your License Here]

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the debug information in the app
3. Open an issue with details about your setup and error messages
