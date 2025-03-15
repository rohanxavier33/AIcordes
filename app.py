import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import librosa
import tempfile
from pytube import YouTube
import shutil
from urllib.request import urlopen
import io
import requests
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Chord Recognition App",
    page_icon="🎵",
    layout="wide"
)

# Title and description
st.title("🎵 Chord Recognition App")
st.markdown("""
This app identifies chords in audio using a deep learning model trained on the McGill Billboard Dataset.
Upload an audio file or provide a YouTube URL to get started.
""")

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Function to simplify chord (from notebook)
def simplify_chord(chord):
    """
    Simplify chord to standard types (major, minor, 7, etc.)
    """
    if chord == 'N' or chord == 'X':
        return chord
    
    # Extract root and quality
    parts = chord.split(':')
    if len(parts) < 2:
        return chord  # Can't parse
    
    root = parts[0]
    quality = parts[1]
    
    # Handle 1/1 chords
    if quality == '1/1':
        return f"{root}"
    
    # Simple chord types that we keep as is
    if quality in ['maj', 'min', '7', 'sus2', 'sus4', 'aug', 'dim', '5']:
        return chord
    
    # Handle half-diminished
    if quality == 'hdim7':
        return f"{root}:m7b5"
    
    # Simplify other chord types
    if 'min' in quality:
        return f"{root}:min"
    
    if 'maj' in quality and '7' not in quality:
        return f"{root}:maj"
    
    if 'maj7' in quality or 'maj9' in quality or 'maj13' in quality:
        return f"{root}:maj"
    
    if '7' in quality or '9' in quality or '13' in quality:
        return f"{root}:7"
    
    if '5' in quality:
        return f"{root}:5"
    
    if 'sus' in quality:
        if 'sus2' in quality:
            return f"{root}:sus2"
        if 'sus4' in quality:
            return f"{root}:sus4"
    
    if 'aug' in quality:
        return f"{root}:aug"
    
    if 'dim' in quality:
        return f"{root}:dim"
    
    # Default to major if we can't determine
    return f"{root}:maj"

# Function to extract chroma features from audio file
def extract_chroma_features(audio_path, sr=44100, hop_length=512):
    """
    Extract chroma features from audio file
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract chroma features - using chroma_cqt for better results
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # Normalize features 
    chroma = librosa.util.normalize(chroma, axis=0)
    
    # Transpose to get (frames × features) shape as expected by model
    chroma = chroma.T
    
    return chroma, duration

# Function to create windowed examples
def create_windowed_examples(features, window_size=30, hop_size=5):
    """
    Create windowed examples from a sequence of features
    """
    num_frames = len(features)
    X_windows = []
    window_centers = []
    
    for i in range(0, num_frames - window_size + 1, hop_size):
        # Extract window of chroma features
        window = features[i:i + window_size]
        X_windows.append(window)
        
        # Calculate center index for timing information
        center_idx = i + window_size // 2
        window_centers.append(center_idx)
    
    return np.array(X_windows), np.array(window_centers)

# Function to predict chords from chroma features
def predict_chords(model, features, label_encoder, window_size=30, hop_size=5):
    """
    Predict chords from chroma features
    """
    # Create windowed examples
    X_windows, window_centers = create_windowed_examples(features, window_size, hop_size)
    
    # Make predictions
    y_pred = model.predict(X_windows)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get prediction confidences
    confidences = np.max(y_pred, axis=1)
    
    # Convert to chord labels
    predictions = label_encoder.inverse_transform(y_pred_classes)
    
    # Calculate timestamps (frames to time)
    hop_length = 512  # Librosa default
    sr = 44100  # Sample rate
    frame_time = hop_length / sr
    timestamps = window_centers * frame_time
    
    return predictions, timestamps, confidences

# Function to smooth predictions
def smooth_predictions(predictions, confidences=None, window_size=5):
    """
    Smooth predictions by majority voting within a window
    """
    smoothed = np.copy(predictions)
    for i in range(len(predictions)):
        start = max(0, i - window_size // 2)
        end = min(len(predictions), i + window_size // 2 + 1)
        window = predictions[start:end]
        
        if confidences is not None:
            window_confidences = confidences[start:end]
            # Weight by confidence
            unique, indices = np.unique(window, return_inverse=True)
            weighted_counts = np.zeros(len(unique))
            for j, idx in enumerate(indices):
                weighted_counts[idx] += window_confidences[j]
            most_common = unique[np.argmax(weighted_counts)]
        else:
            # Simple majority voting
            unique, counts = np.unique(window, return_counts=True)
            most_common = unique[np.argmax(counts)]
        
        smoothed[i] = most_common
    
    return smoothed

# Function to download and extract audio from YouTube video
def download_youtube_audio(youtube_url):
    """
    Download and extract audio from YouTube video using pytube
    (no FFmpeg required)
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'audio.mp4')
    
    try:
        # Download YouTube video audio stream
        st.info("Connecting to YouTube...")
        yt = YouTube(youtube_url)
        video_title = yt.title
        
        st.info(f"Downloading audio from: {video_title}")
        # Get the audio stream with highest quality
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        
        if not audio_stream:
            raise Exception("No audio stream found for this video")
        
        # Download the file
        audio_file = audio_stream.download(output_path=temp_dir)
        
        # Rename to temp_file path if needed
        if audio_file != temp_file:
            shutil.move(audio_file, temp_file)
        
        return temp_file, video_title
        
    except Exception as e:
        st.error(f"Error downloading YouTube video: {str(e)}")
        raise e

# Function to load model and label encoder
@st.cache_resource
def load_pretrained_model(model_path, encoder_path):
    """
    Load pretrained model and label encoder
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure model and encoder files exist at the specified paths.")
        return None, None

# Function to visualize predictions
def visualize_predictions(predictions, timestamps, confidences=None, title="Chord Predictions", time_limit=None):
    """
    Visualize chord predictions
    """
    # Filter by time limit if specified
    if time_limit is not None:
        mask = timestamps <= time_limit
        predictions = predictions[mask]
        timestamps = timestamps[mask]
        if confidences is not None:
            confidences = confidences[mask]
    
    # Get unique chord labels for color mapping
    unique_chords = np.unique(predictions)
    color_map = {chord: i for i, chord in enumerate(unique_chords)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot segments
    prev_time = timestamps[0]
    prev_chord = predictions[0]
    prev_confidence = confidences[0] if confidences is not None else None
    
    for i in range(1, len(timestamps)):
        current_time = timestamps[i]
        current_chord = predictions[i]
        current_confidence = confidences[i] if confidences is not None else None
        
        if current_chord != prev_chord:
            # Plot segment
            ax.axvspan(prev_time, current_time, 
                       alpha=0.3, 
                       color=plt.cm.tab20(color_map[prev_chord] % 20))
            
            # Add chord label at segment midpoint
            mid_time = (prev_time + current_time) / 2
            label_text = prev_chord
            if prev_confidence is not None:
                label_text += f"\n{prev_confidence:.2f}"
                
            ax.text(mid_time, 0.5, label_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10)
            
            prev_time = current_time
            prev_chord = current_chord
            prev_confidence = current_confidence
    
    # Plot final segment
    ax.axvspan(prev_time, timestamps[-1], 
               alpha=0.3, 
               color=plt.cm.tab20(color_map[prev_chord] % 20))
    
    # Add final chord label
    mid_time = (prev_time + timestamps[-1]) / 2
    label_text = prev_chord
    if prev_confidence is not None:
        label_text += f"\n{prev_confidence:.2f}"
        
    ax.text(mid_time, 0.5, label_text, 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10)
    
    # Configure plot
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Return figure
    return fig

# Function to trim audio file if needed
def trim_audio(audio_path, max_duration=300):
    """
    Trim audio file to a maximum duration
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Check if trimming is needed
    if duration <= max_duration:
        return audio_path
    
    # Trim audio
    y_trimmed = y[:int(max_duration * sr)]
    
    # Create temporary file for trimmed audio
    trimmed_path = os.path.join(tempfile.mkdtemp(), 'trimmed_audio.wav')
    
    # Use soundfile instead of librosa.output.write_wav (which is deprecated)
    import soundfile as sf
    sf.write(trimmed_path, y_trimmed, sr)
    
    return trimmed_path

# Create Streamlit UI
st.sidebar.title("Options")

# Input source selection
input_source = st.sidebar.radio("Select input source:", ("Upload Audio File", "YouTube URL"))

# Model paths
model_path = st.sidebar.text_input("Model path:", "models/best_chord_model.keras")
encoder_path = st.sidebar.text_input("Encoder path:", "models/chord_recognition_encoder.pkl")

# Advanced options
with st.sidebar.expander("Advanced Options"):
    window_size = st.number_input("Window size:", min_value=5, max_value=100, value=30)
    hop_size = st.number_input("Hop size:", min_value=1, max_value=20, value=5)
    smoothing_window = st.number_input("Smoothing window size:", min_value=1, max_value=20, value=5)
    time_limit = st.number_input("Time limit (seconds, 0 for no limit):", min_value=0, value=0)
    time_limit = None if time_limit == 0 else time_limit
    max_duration = st.number_input("Maximum audio duration (seconds):", min_value=30, max_value=600, value=300)

# Main area
if input_source == "Upload Audio File":
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac"])
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file)
        
        # Process uploaded file
        with st.spinner("Processing audio file..."):
            # Save uploaded file to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_path = tmp.name
            
            try:
                # Trim audio if needed
                audio_path = trim_audio(audio_path, max_duration)
                
                # Show progress bar
                progress_bar = st.progress(0)
                
                # Extract chroma features
                progress_bar.progress(10)
                st.info("Extracting chroma features...")
                chroma_features, duration = extract_chroma_features(audio_path)
                
                # Load model and label encoder
                progress_bar.progress(30)
                st.info("Loading model...")
                model, label_encoder = load_pretrained_model(model_path, encoder_path)
                
                if model is None or label_encoder is None:
                    st.error("Failed to load model or label encoder.")
                else:
                    # Predict chords
                    progress_bar.progress(50)
                    st.info("Predicting chords...")
                    predictions, timestamps, confidences = predict_chords(
                        model, chroma_features, label_encoder, window_size, hop_size
                    )
                    
                    # Smooth predictions
                    progress_bar.progress(70)
                    st.info("Smoothing predictions...")
                    if smoothing_window > 1:
                        predictions = smooth_predictions(predictions, confidences, smoothing_window)
                    
                    # Visualize predictions
                    progress_bar.progress(90)
                    st.info("Generating visualization...")
                    fig = visualize_predictions(
                        predictions, timestamps, confidences,
                        title=f"Chord Predictions - {uploaded_file.name}",
                        time_limit=time_limit
                    )
                    st.pyplot(fig)
                    
                    # Create a downloadable chord chart
                    chord_data = pd.DataFrame({
                        'Time (s)': timestamps,
                        'Chord': predictions,
                        'Confidence': confidences
                    })
                    
                    # Show data table
                    st.subheader("Chord Predictions")
                    st.dataframe(chord_data)
                    
                    # Convert to CSV for download
                    csv = chord_data.to_csv(index=False)
                    st.download_button(
                        label="Download Chord Chart",
                        data=csv,
                        file_name=f"{uploaded_file.name.split('.')[0]}_chords.csv",
                        mime="text/csv"
                    )
                
                progress_bar.progress(100)
                
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
            
            finally:
                # Clean up temporary file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

else:  # YouTube URL
    youtube_url = st.text_input("Enter YouTube URL:")
    
    if youtube_url:
        # Check if it's a valid YouTube URL
        if "youtube.com" in youtube_url or "youtu.be" in youtube_url:
            with st.spinner("Downloading YouTube video..."):
                try:
                    # Download and extract audio from YouTube video
                    audio_path, video_title = download_youtube_audio(youtube_url)
                    
                    # Display audio player
                    audio_data = open(audio_path, 'rb').read()
                    st.audio(audio_data)
                    
                    # Show progress bar
                    progress_bar = st.progress(0)
                    
                    # Trim audio if needed
                    audio_path = trim_audio(audio_path, max_duration)
                    
                    # Extract chroma features
                    progress_bar.progress(10)
                    st.info("Extracting chroma features...")
                    chroma_features, duration = extract_chroma_features(audio_path)
                    
                    # Load model and label encoder
                    progress_bar.progress(30)
                    st.info("Loading model...")
                    model, label_encoder = load_pretrained_model(model_path, encoder_path)
                    
                    if model is None or label_encoder is None:
                        st.error("Failed to load model or label encoder.")
                    else:
                        # Predict chords
                        progress_bar.progress(50)
                        st.info("Predicting chords...")
                        predictions, timestamps, confidences = predict_chords(
                            model, chroma_features, label_encoder, window_size, hop_size
                        )
                        
                        # Smooth predictions
                        progress_bar.progress(70)
                        st.info("Smoothing predictions...")
                        if smoothing_window > 1:
                            predictions = smooth_predictions(predictions, confidences, smoothing_window)
                        
                        # Visualize predictions
                        progress_bar.progress(90)
                        st.info("Generating visualization...")
                        fig = visualize_predictions(
                            predictions, timestamps, confidences,
                            title=f"Chord Predictions - {video_title}",
                            time_limit=time_limit
                        )
                        st.pyplot(fig)
                        
                        # Create a downloadable chord chart
                        chord_data = pd.DataFrame({
                            'Time (s)': timestamps,
                            'Chord': predictions,
                            'Confidence': confidences
                        })
                        
                        # Show data table
                        st.subheader("Chord Predictions")
                        st.dataframe(chord_data)
                        
                        # Convert to CSV for download
                        csv = chord_data.to_csv(index=False)
                        st.download_button(
                            label="Download Chord Chart",
                            data=csv,
                            file_name=f"youtube_chords.csv",
                            mime="text/csv"
                        )
                    
                    progress_bar.progress(100)
                    
                except Exception as e:
                    st.error(f"Error processing YouTube video: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if 'audio_path' in locals() and os.path.exists(audio_path):
                        os.unlink(audio_path)
        else:
            st.error("Invalid YouTube URL")

# Add info about the model
with st.expander("About the Model"):
    st.markdown("""
    ### Chord Recognition Model
    
    This app uses a deep learning model trained on the McGill Billboard Dataset to recognize chords in music.
    
    The model architecture is a hybrid CNN-LSTM with attention mechanisms and residual connections, as described
    in the notebook. It takes chroma features as input and outputs chord predictions.
    
    **Features:**
    - Can recognize major, minor, dominant 7th, and other chord types
    - Processes audio using chroma features
    - Works with uploaded audio files or YouTube videos
    
    **Limitations:**
    - Accuracy may vary depending on audio quality and complexity
    - May struggle with unusual chord voicings or extended harmonies
    - Model was trained on popular music, so results may be less accurate for other genres
    """)

# Add installation instructions
with st.expander("Installation Instructions"):
    st.markdown("""
    ### Required Packages
    
    To run this app, you'll need to install the following packages:
    
    ```
    pip install streamlit numpy pandas matplotlib tensorflow librosa yt-dlp scikit-learn
    ```
    
    ### Model Files
    
    You'll need to have the trained model files available:
    
    1. The trained model file (`best_chord_model.keras`) in the `models` directory
    2. The label encoder file (`chord_recognition_encoder.pkl`) in the `models` directory
    
    ### Running the App
    
    Save this script as `app.py` and run:
    
    ```
    streamlit run app.py
    ```
    """)

# Add footer
st.markdown("---")
st.markdown("Chord Recognition App | McGill Billboard Dataset")