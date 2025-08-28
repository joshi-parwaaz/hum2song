"""
app.py - Streamlit web application for the hum2song project.
This app allows users to upload audio files (songs or humming) and recognize them.
"""

import os
import sys
import json
import tempfile
import streamlit as st
from pathlib import Path

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.recognize import recognize_song, format_results
from scripts.ingest import ingest_folder, load_song_mapping
from utils.audio_utils import load_audio

# Define default paths
DEFAULT_DB_PATH = os.path.join(project_root, "data", "fingerprints.db")
DEFAULT_MAPPING_PATH = os.path.join(project_root, "data", "song_mapping.json")
DEFAULT_FEATURE_DIR = os.path.join(project_root, "data", "features")

# Create directories if they don't exist
os.makedirs(os.path.dirname(DEFAULT_DB_PATH), exist_ok=True)
os.makedirs(DEFAULT_FEATURE_DIR, exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Hum2Song - Song Recognition",
    page_icon="🎵",
    layout="wide"
)

# Import webrtc components
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av
    import queue
    import threading
    import time
    import wave
    import io
    import numpy as np
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("streamlit_webrtc is not installed. Microphone recording will not be available.")

# Application title and description
st.title("Hum2Song: Song Recognition")
st.markdown("""
This application can recognize songs from your humming.
- **Upload**: Upload an audio file or select a sample
- **Record**: Use the microphone to hum a song (if available)
- **Recognize**: The system will try to identify the song
""")

# Load song mapping to check if we have songs in the database
song_mapping = load_song_mapping(DEFAULT_MAPPING_PATH)
song_count = len(song_mapping) if song_mapping else 0

if song_count == 0:
    st.warning("No songs in the database. Please add songs using the ingest script first.")
    st.markdown("""
    **To get started:**
    1. Place your MP3/WAV files in the `data/songs_mp3/` directory
    2. Run: `python scripts/ingest.py data/songs_mp3`
    3. Refresh this page
    """)
else:
    st.info(f"Currently {song_count} songs in the database.")

# Add an option to use an existing audio file
st.markdown("### 🎵 Upload an audio file or select a sample")
uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3"])

# Add a sample file selection
song_dir = os.path.join(project_root, "data", "songs_mp3")
sample_path = None
if os.path.exists(song_dir):
    sample_files = [f for f in os.listdir(song_dir) if f.lower().endswith(('.mp3', '.wav'))]
    if sample_files:
        sample_options = ["Select a sample file..."] + sample_files
        selected_sample = st.selectbox("Or choose a sample file", sample_options)
        
        if selected_sample != "Select a sample file...":
            sample_path = os.path.join(song_dir, selected_sample)
            st.session_state["audio_file_path"] = sample_path
            st.success(f"Using sample file: {selected_sample}")
            
            # Display the sample audio
            with open(sample_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
else:
    st.info("No songs directory found. Create 'data/songs_mp3/' and add your songs there.")

if uploaded_file is not None:
    # Save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        audio_file_path = tmp_file.name
    
    st.session_state["audio_file_path"] = audio_file_path
    st.success("Audio file uploaded successfully!")
    
    # Display the uploaded audio
    st.audio(uploaded_file, format="audio/wav")

# WebRTC audio recording (if available)
if WEBRTC_AVAILABLE:
    st.markdown("### 🎤 Or record your humming")
    
    # Create a class for audio recording with WebRTC
    class AudioProcessor:
        def __init__(self):
            self.audio_frames = []
            self.recording = False
            self.recorded_audio = None
            self.recorded_file_path = None
            
        def process_audio(self, frame):
            if self.recording:
                # Store audio frames while recording
                self.audio_frames.append(frame.to_ndarray())
            return frame
            
        def save_audio(self):
            # Convert recorded audio frames to WAV file
            if not self.audio_frames:
                return None
                
            # Create a temporary file for the recording
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Save audio as WAV file
            audio_data = self.audio_frames[0]
            sample_rate = 48000  # WebRTC default sample rate
            
            with wave.open(temp_file_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                
                for frame in self.audio_frames:
                    # Convert to int16 format for WAV
                    wf.writeframes((frame * 32767).astype(np.int16).tobytes())
            
            self.recorded_file_path = temp_file_path
            return temp_file_path
    
    # Create an instance of the audio processor
    audio_processor = AudioProcessor()
    
    # Columns for record and stop buttons
    col1, col2 = st.columns(2)
    
    # Record button
    if col1.button("Start Recording"):
        audio_processor.recording = True
        audio_processor.audio_frames = []
        st.session_state["recording"] = True
        st.info("Recording started... Hum your song!")
    
    # Stop button
    if col2.button("Stop Recording"):
        if st.session_state.get("recording", False):
            audio_processor.recording = False
            st.session_state["recording"] = False
            
            # Save the recorded audio
            audio_file_path = audio_processor.save_audio()
            if audio_file_path:
                st.session_state["audio_file_path"] = audio_file_path
                st.success("Recording saved! Ready to recognize.")
            else:
                st.warning("No audio was recorded. Please try again.")
    
    # WebRTC streamer for audio
    try:
        webrtc_ctx = webrtc_streamer(
            key="humming",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            video_processor_factory=None,
            audio_processor_factory=lambda: audio_processor,
        )
    except Exception as e:
        st.error(f"WebRTC error: {str(e)}")
    
    # Get audio file path from session state if available
    audio_file_path = st.session_state.get("audio_file_path", None)
    
    # Display the recorded audio if available
    if audio_file_path and os.path.exists(audio_file_path) and audio_file_path.endswith(".wav"):
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
        
        st.info("Your humming is ready for recognition!")

# Recognition section - only show if we have songs in database
if song_count > 0:
    # Recognition parameters
    st.markdown("### ⚙️ Recognition Settings")
    col1, col2 = st.columns(2)
    with col1:
        fingerprint_threshold = st.slider("Fingerprint Confidence Threshold", 0.0, 1.0, 0.05, 0.01, 
                                         help="Lower values will match more songs but with less confidence")
    with col2:
        dtw_threshold = st.slider("DTW Similarity Threshold", 0.0, 1.0, 0.3, 0.01,
                                 help="Lower values will match more humming patterns but with less confidence")

    max_results = st.slider("Maximum Results", 1, 10, 5, 
                           help="Maximum number of matching songs to display")

    # Process the audio file
    audio_available = False
    tmp_path = None

    if "audio_file_path" in st.session_state and os.path.exists(st.session_state["audio_file_path"]):
        # Use the audio file
        tmp_path = st.session_state["audio_file_path"]
        audio_available = True
        
        # Run recognition
        if audio_available and st.button("🎵 Recognize Song"):
            with st.spinner("Analyzing audio..."):
                try:
                    # Check if audio can be loaded
                    try:
                        audio, sr = load_audio(tmp_path)
                        st.success(f"Audio loaded successfully: {sr}Hz, {len(audio)/sr:.2f} seconds")
                    except Exception as e:
                        st.error(f"Error loading audio: {str(e)}")
                        st.stop()
                    
                    # Run recognition
                    results = recognize_song(
                        tmp_path, 
                        db_path=DEFAULT_DB_PATH,
                        mapping_path=DEFAULT_MAPPING_PATH,
                        feature_dir=DEFAULT_FEATURE_DIR,
                        fingerprint_threshold=fingerprint_threshold,
                        dtw_threshold=dtw_threshold,
                        max_results=max_results
                    )
                    
                    # Display results
                    if results.get('success', False):
                        st.success("Recognition completed!")
                        
                        # Display best match
                        best_match = results.get('best_match')
                        if best_match:
                            st.subheader("🎯 Best Match")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Song", f"{best_match['artist']} - {best_match['title']}")
                                st.metric("Match Type", best_match['match_type'].capitalize())
                            with col2:
                                fp_confidence = best_match['fingerprint_confidence'] * 100
                                dtw_similarity = best_match['dtw_similarity'] * 100
                                st.metric("Fingerprint Confidence", f"{fp_confidence:.1f}%")
                                st.metric("DTW Similarity", f"{dtw_similarity:.1f}%")
                        
                        # Display alternatives
                        alternatives = results.get('alternatives', [])
                        if alternatives:
                            st.subheader("🎼 Alternative Matches")
                            for i, alt in enumerate(alternatives):
                                with st.expander(f"#{i+1}: {alt['artist']} - {alt['title']}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Match Type", alt['match_type'].capitalize())
                                    with col2:
                                        fp_confidence = alt['fingerprint_confidence'] * 100
                                        dtw_similarity = alt['dtw_similarity'] * 100
                                        st.metric("Fingerprint Confidence", f"{fp_confidence:.1f}%")
                                        st.metric("DTW Similarity", f"{dtw_similarity:.1f}%")
                    else:
                        st.error(f"Recognition failed: {results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error during recognition: {str(e)}")
                
                # Add debug output
                with st.expander("🔍 Debug Information"):
                    st.write("Raw Recognition Results:")
                    st.json(results)
                    
                    # Display additional information about the song mapping
                    st.write("Available Songs in Database:")
                    for song_id, song_info in song_mapping.items():
                        st.write(f"ID: {song_id}, Title: {song_info.get('title')}, Artist: {song_info.get('artist')}")
                
                # Clean up the temporary file if it was uploaded (not a sample file)
                if uploaded_file is not None and tmp_path != sample_path:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

# Add some information at the bottom
st.markdown("---")
st.markdown("## How it Works")
st.markdown("""
The hum2song system uses a two-tiered approach to recognize songs:

1. **Audio Fingerprinting**: Fast and accurate for exact audio matches
2. **Dynamic Time Warping (DTW)**: More flexible for recognizing hummed or sung variations

For best results:
- When humming, try to match the melody as closely as possible
- Record in a quiet environment
- Use high-quality audio files when adding songs to the database
""")

# Sidebar with additional information
with st.sidebar:
    st.title("About Hum2Song")
    st.markdown("""
    Hum2Song is a song recognition system designed to recognize both exact audio matches and hummed/sung variations.
    """)
    
    # Display database stats
    st.subheader("Database Statistics")
    st.metric("Songs in Database", song_count)
    
    # Add a button to check for database updates
    if st.button("Refresh Database Stats"):
        st.experimental_rerun()
        
    # Add setup instructions
    st.subheader("Setup Instructions")
    st.markdown("""
    **To add songs to the database:**
    1. Place MP3/WAV files in `data/songs_mp3/`
    2. Run: `python scripts/download_songs.py` (optional - downloads samples)
    3. Run: `python scripts/ingest.py data/songs_mp3`
    4. Refresh this page
    
    **File structure:**
    ```
    hum2song/
    ├── data/
    │   ├── songs_mp3/     # Put your songs here
    │   ├── features/      # Auto-generated
    │   └── fingerprints.db # Auto-generated
    ├── scripts/
    └── utils/
    ```
    """)
