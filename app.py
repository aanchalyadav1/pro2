import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ------------------ SPOTIFY SETUP ------------------
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# ------------------ UI SETUP ------------------
st.set_page_config(page_title="Moodify Music", page_icon="🎵", layout="centered")
st.title("🎵 Moodify Music Recommendations (Based on Your Mood)")
st.write("Upload a selfie or any image of a face, and get music recommendations based on detected emotion!")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    rgb_image = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing emotion..."):
        try:
            result = DeepFace.analyze(rgb_image, actions=['emotion'], enforce_detection=False)
            emotion_detected = result[0]['dominant_emotion']
            st.success(f"**Detected Emotion:** {emotion_detected.capitalize()}")
        except Exception as e:
            st.error(f"Emotion detection failed: {e}")
            emotion_detected = None

    if emotion_detected:
        # Map emotion to genre
        emotion_to_genre = {
            "happy": "pop",
            "sad": "acoustic",
            "angry": "metal",
            "surprise": "edm",
            "fear": "ambient",
            "disgust": "punk",
            "neutral": "lofi"
        }

        genre = emotion_to_genre.get(emotion_detected.lower(), "pop")

        st.subheader(f"🎶 Music Recommendations for {emotion_detected.capitalize()} mood:")

        try:
            results = sp.search(q=f"genre:{genre}", type="track", limit=5)
            for idx, track in enumerate(results['tracks']['items'], 1):
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                spotify_url = track['external_urls']['spotify']
                st.write(f"{idx}. [{track_name} by {artist_name}]({spotify_url})")
        except Exception as e:
            st.error(f"Spotify error: {e}")
else:
    st.info("👆 Upload an image to get started!")
