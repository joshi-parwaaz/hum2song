# Quick Start Guide for Hum2Song

## For New Users with Their Own Dataset

### Step 1: Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup check
python setup.py --check-only
```

### Step 2: Add Your Songs
1. Create the songs directory (if it doesn't exist):
   ```bash
   mkdir -p data/songs_mp3
   ```

2. Copy your MP3/WAV files to `data/songs_mp3/`
   - Name them like: `Artist - Song Title.mp3`
   - Supported formats: MP3, WAV, FLAC, OGG

### Step 3: Process Your Songs
```bash
# Option 1: Automatic setup (recommended)
python setup.py

# Option 2: Manual processing
python scripts/ingest.py data/songs_mp3
```

### Step 4: Run the App
```bash
streamlit run app.py
```

## What Gets Created Automatically

When you run the ingest script, these files are created:
- `data/fingerprints.db` - SQLite database with song fingerprints
- `data/song_mapping.json` - JSON file mapping song IDs to metadata  
- `data/features/` - Directory with extracted audio features (.npy files)

## Troubleshooting

### No Songs Recognized?
- Check that songs were properly ingested: look for files in `data/features/`
- Lower the recognition thresholds in the app
- Check the debug information in the app's expandable section

### Import Errors?
- Run `python setup.py --check-only` to verify dependencies
- Install missing packages: `pip install -r requirements.txt`

### Audio Loading Issues?
- Install ffmpeg: `pip install ffmpeg-python`
- Ensure your audio files are valid MP3/WAV format

## File Structure After Setup

```
hum2song/
├── data/
│   ├── songs_mp3/          # Your songs here
│   ├── features/           # Auto-generated (.npy files)
│   ├── fingerprints.db     # Auto-generated database
│   └── song_mapping.json   # Auto-generated mapping
├── scripts/
├── utils/
├── app.py                  # Run this with streamlit
├── setup.py               # Setup helper
└── requirements.txt       # Dependencies
```

## Using the App

1. **Upload Audio**: Use file uploader or select from samples
2. **Adjust Settings**: Lower thresholds for more matches
3. **Recognize**: Click the recognize button
4. **View Results**: See match confidence and alternatives

The app shows both fingerprint confidence (exact matches) and DTW similarity (humming matches).

## Command Line Usage

```bash
# Download sample songs (optional)
python scripts/download_songs.py --max 10

# Process specific directory
python scripts/ingest.py path/to/your/songs

# Recognize single file
python scripts/recognize.py path/to/audio/file.wav
```

## Tips for Best Results

- Use high-quality audio files for better fingerprinting
- Name files clearly: "Artist - Song Title.mp3"
- For humming: match the melody as closely as possible
- Start with lower threshold values if not getting matches
