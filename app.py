
import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

# Spotify setup
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# App UI
st.set_page_config(page_title="Moodify Music", page_icon="🎵")
st.title("🎵 Moodify Music Recommendations (Live Webcam)")
st.write("Look into your webcam and get music recommendations based on your mood!")

# Webcam capture
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
emotion_display = st.empty()
last_analysis_time = 0
emotion_detected = None
ANALYSIS_INTERVAL = 3  # seconds

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(rgb_frame)

    current_time = time.time()
    if current_time - last_analysis_time > ANALYSIS_INTERVAL:
        try:
            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotion_detected = result['dominant_emotion']
            emotion_display.markdown(f"### Detected Emotion: **{emotion_detected.capitalize()}**")
            last_analysis_time = current_time
        except:
            emotion_display.warning("Unable to detect emotion.")

    if st.button("Get Music Recommendations"):
        if emotion_detected:
            emotion_to_genre = {
                "happy": "pop",
                "sad": "acoustic",
                "angry": "metal",
                "surprise": "edm",
                "fear": "ambient",
                "disgust": "punk",
                "neutral": "lofi"
            }
            genre = emotion_to_genre.get(emotion_detected, "pop")
            results = sp.search(q=f"genre:{genre}", type="track", limit=5)

            st.subheader("🎶 Recommended Tracks:")
            for idx, track in enumerate(results['tracks']['items'], 1):
                track_name = track['name']
                artist_name = track['artists'][0]['name']
                spotify_url = track['external_urls']['spotify']
                st.write(f"{idx}. [{track_name} by {artist_name}]({spotify_url})")
        else:
            st.warning("Emotion not detected yet. Please look at the camera.")

cap.release()
