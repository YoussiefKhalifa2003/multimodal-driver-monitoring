# fusion_driver_demo.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from simulated_state_generator import SimulatedStateGenerator
import sys
import random
import spotipy
import time
import os
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# === Emotion to Playlist Mapping === 
EMOTION_PLAYLIST = {
    "angry": "Rock",
    "happy": "EDM",
    "sad": "Soul"
}

# Updated emotion labels - ensure this matches the model output and dataset
EMOTION_LABELS = ['happy', 'sad', 'angry']

# Print model information before attempting to load
print("Attempting to load model...")

# Load facial emotion model
try:
    emotion_model = load_model('facial_emotion_model.h5')
    print("Facial emotion model loaded successfully")
    print(f"Input shape: {emotion_model.input_shape}")
    print(f"Output shape: {emotion_model.output_shape}")
except Exception as e:
    print(f"Error loading facial emotion model: {e}")
    sys.exit(1)
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# For eye aspect ratio (EAR) calculation
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

def shape_to_np(shape):
    return np.array([[p[0], p[1]] for p in shape], dtype="int")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Use dlib for facial landmarks (if available)
try:
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(cv2.data.haarcascades + '../shape_predictor_68_face_landmarks.dat')
    dlib_available = True
except Exception as e:
    print("dlib not available, EAR-based graph will be random.")
    dlib_available = False

def draw_emotion_graph(frame, signal, top_left=None, size=(200, 80)):
    if top_left is None:
        top_left = (frame.shape[1] - 220, 20)
    x0, y0 = top_left
    w, h = size
    cv2.rectangle(frame, (x0, y0), (x0+w, y0+h), (0,0,0), 1)
    if len(signal) < 2:
        return
    norm_signal = (np.array(signal) - np.min(signal)) / (np.ptp(signal) + 1e-6)
    for i in range(len(signal)-1):
        x1 = x0 + int(i * w / (len(signal)-1))
        y1 = y0 + h - int(norm_signal[i] * h)
        x2 = x0 + int((i+1) * w / (len(signal)-1))
        y2 = y0 + h - int(norm_signal[i+1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

# Camera selection: try built-in webcam first
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Default webcam (index 0) not found, trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open any webcam!")
        sys.exit(1)
    else:
        print("Using external/phone camera (index 1)")
else:
    print("Using built-in laptop webcam (index 0)")

# Set camera properties for better face detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Webcam initialized successfully!")
print("Auto mode enabled: Emotion detection every 10 seconds")
print("Press 'm' to toggle auto/manual mode, SPACE to capture in manual mode, ESC to exit.")

sim_state = SimulatedStateGenerator(interval=5)
sim_state.start()
last_music_state = None

# Add emotion intensity graph logic
EMOTION_INTENSITY = {
    'happy': 0.8,
    'sad': 0.3,
    'angry': 0.9,
    'neutral': 0.5
}

emotion_signal = [0.5 for _ in range(60)]  # Start with neutral

# Update Spotify playlists for new emotions
SPOTIFY_PLAYLISTS = {
    "happy": "spotify:playlist:37i9dQZF1DXdPec7aLTmlC",  # Happy playlist
    "sad": "https://open.spotify.com/playlist/37i9dQZF1EIcqv6dNT3Dgk?si=0dd50d8b815c4ad2",  # Sad playlist
    "angry": "https://open.spotify.com/playlist/0vvXsWCC9xrXsKd4FyS8kM?si=7349d49617d24210"  # Angry playlist
}

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
    redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI', 'http://127.0.0.1:8888/callback'),
    scope='user-modify-playback-state user-read-playback-state user-read-currently-playing streaming'
))

def play_spotify_playlist(emotion):
    try:
        # First check if we can get devices
        devices = sp.devices()
        print("Available devices:", devices)  # Debug print
        
        if not devices['devices']:
            print('No active Spotify device found. Please open Spotify on your computer or phone.')
            return
            
        # Find the desktop app device
        desktop_device = None
        for device in devices['devices']:
            if 'MacBook' in device['name'] or 'Desktop' in device['name']:
                desktop_device = device
                break
                
        if not desktop_device:
            print('Desktop app not found. Please make sure Spotify desktop app is open.')
            return
            
        device_id = desktop_device['id']
        print(f"Using desktop device: {desktop_device['name']}")  # Debug print
        
        # Get the playlist URI
        playlist_uri = SPOTIFY_PLAYLISTS.get(emotion)
        if not playlist_uri:
            print(f"No playlist defined for emotion: {emotion}")
            return
            
        print(f"Attempting to play playlist: {playlist_uri}")  # Debug print
        
        # First transfer playback to desktop app
        sp.transfer_playback(device_id=device_id, force_play=True)
        print("Transferred playback to desktop app")
        
        # Then start playback
        sp.start_playback(device_id=device_id, context_uri=playlist_uri)
        print(f"Successfully started playback for {emotion}")
        
    except spotipy.exceptions.SpotifyException as e:
        print(f"Spotify Error: {str(e)}")
        if "PREMIUM_REQUIRED" in str(e):
            print("Please make sure:")
            print("1. You are logged into Spotify with your Premium account")
            print("2. You have an active Spotify device (desktop app or web player)")
            print("3. Your Premium subscription is active")
        else:
            print(f"Other Spotify error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Auto-detection settings
auto_mode = True  # Start in auto mode
emotion_check_interval = 10  # Check emotion every 10 seconds
last_emotion_check_time = time.time()
current_emotion = None
last_detected_emotion = None
stable_emotion_counter = 0
required_stable_frames = 5  # Number of consistent detections needed to change music

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    frame = cv2.flip(frame, 1)  # Mirror the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try different scale factors for face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    webcam_state = "neutral"
    box = None
    probs = None

    # Process detected faces
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            if w > 30 and h > 30:  # Only process faces of reasonable size
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract and preprocess face
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (64, 64))
                face_img = face_img.astype('float32') / 255.0
                face_img = np.expand_dims(face_img, axis=-1)
                face_img = np.expand_dims(face_img, axis=0)
                
                try:
                    # Get emotion prediction
                    preds = emotion_model.predict(face_img, verbose=0)[0]
                    pred_index = np.argmax(preds)
                    
                    if pred_index < len(EMOTION_LABELS):
                        webcam_state = EMOTION_LABELS[pred_index]
                    else:
                        webcam_state = "neutral"
                        
                    box = (x, y, w, h)
                    probs = preds
                    
                    # Update current emotion for auto mode
                    current_emotion = webcam_state
                    
                    # Check if emotion is stable
                    if current_emotion == last_detected_emotion:
                        stable_emotion_counter += 1
                    else:
                        stable_emotion_counter = 0
                        last_detected_emotion = current_emotion
                    
                    # Draw emotion label
                    cv2.putText(frame, webcam_state, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    
                    # Draw probability table
                    table_x, table_y = x, y + h + 10
                    table_w, table_h = 180, 20 * (len(EMOTION_LABELS) + 1)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (table_x, table_y), (table_x + table_w, table_y + table_h), (0, 0, 0), -1)
                    alpha = 0.5
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    cv2.putText(frame, "Face", (table_x + 10, table_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    
                    for i, emotion in enumerate(EMOTION_LABELS):
                        if i < len(preds):
                            cv2.putText(frame, f"{emotion}: {preds[i]:.2f}", 
                                      (table_x + 10, table_y + 40 + 20*i), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
                break
    else:
        # If no face detected, show message
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        current_emotion = None
                
    # Update emotion signal for graph
    intensity = EMOTION_INTENSITY.get(webcam_state, 0.5)
    emotion_signal.append(emotion_signal[-1]*0.8 + intensity*0.2)
    if len(emotion_signal) > 60:
        emotion_signal.pop(0)

    # Auto emotion check and music change
    current_time = time.time()
    if auto_mode and (current_time - last_emotion_check_time) >= emotion_check_interval:
        last_emotion_check_time = current_time
        
        if current_emotion is not None and stable_emotion_counter >= required_stable_frames:
            if current_emotion != last_music_state:
                print(f"Auto detected: {current_emotion.upper()} → {EMOTION_PLAYLIST.get(current_emotion, 'Unknown')}")
                play_spotify_playlist(current_emotion)
                last_music_state = current_emotion
                
                # Show status message
                cv2.putText(frame, f"Changed music to: {EMOTION_PLAYLIST.get(current_emotion, 'Unknown')}", 
                           (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                # Show status message that emotion hasn't changed
                cv2.putText(frame, f"Emotion unchanged: {current_emotion}", 
                           (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            # Show status message that not enough stable frames
            cv2.putText(frame, "Waiting for stable emotion detection...", 
                       (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Display mode status
    mode_text = "AUTO MODE" if auto_mode else "MANUAL MODE"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    
    # Display time until next check in auto mode
    if auto_mode:
        time_left = max(0, emotion_check_interval - (current_time - last_emotion_check_time))
        cv2.putText(frame, f"Next check in: {time_left:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    # Display current emotion status
    if current_emotion:
        cv2.putText(frame, f"Current emotion: {current_emotion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    # Display instructions based on mode
    if auto_mode:
        cv2.putText(frame, "Press 'm' to switch to manual mode", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    else:
        cv2.putText(frame, "Press SPACE to capture emotion, 'm' for auto mode", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Draw the emotion intensity graph
    draw_emotion_graph(frame, emotion_signal)

    # Show frame
    cv2.imshow("Fusion Emotion + Music System", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit
        break
    elif key == ord('m'):  # 'm' to toggle auto/manual mode
        auto_mode = not auto_mode
        if auto_mode:
            print("Switched to AUTO mode - emotion detection every 10 seconds")
            last_emotion_check_time = time.time()  # Reset timer
        else:
            print("Switched to MANUAL mode - press SPACE to capture emotion")
    elif key == 32 and not auto_mode:  # SPACE to capture in manual mode
        if current_emotion is not None:
            print(f"Manually captured: {current_emotion.upper()} → {EMOTION_PLAYLIST.get(current_emotion, 'Unknown')}")
            play_spotify_playlist(current_emotion)
            last_music_state = current_emotion

cap.release()
cv2.destroyAllWindows()
sim_state.stop()
print("Session Ended.")