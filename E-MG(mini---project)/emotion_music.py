# Emotion-to-Tamil-Music Generator 
# Developed using OpenCV, DeepFace, and Spotify Playlists

import cv2
from deepface import DeepFace
import webbrowser
import time

# -----------------------------
# Step 1: Emotion-to-Playlist Mapping (Tamil Playlists)
# -----------------------------

emotion_playlists = {
    "happy": "https://open.spotify.com/playlist/4z0Nfo8p5sfNHs9j60NLrp?si=uKh3fyaZTQqPjpGkl0_ucw",    
    "sad": "https://open.spotify.com/playlist/0CiE4CYWUmJ8nyPV5eZJtd?si=3vylM1R6TIy6wXKEWlPuOg&pi=9M5-6h3tT06aj",      
    "angry": "https://open.spotify.com/playlist/2MznoblTybLJagH1BgNPrv?si=eS8orihKTkiZXyci7zzMZw&pi=VgZZLzS3SYOQq",    
    "neutral": "https://open.spotify.com/playlist/3rLhdyd2I8OJDT2HepdgW9?si=nWho2UY6RXa_DmADclbxUQ", 
    "surprise": "https://open.spotify.com/playlist/7q6eqPQthpg1rUbRHj9KMu?si=XPIKM4fnQ-C9JHwAILM8RQ"  
}

# -----------------------------
# Step 2: Start Webcam & Detect Emotion
# -----------------------------

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Detecting emotion... Press 'q' to quit.")

last_emotion = None
last_check_time = 0
emotion_check_interval = 60  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        current_time = time.time()

        # Only check emotion every 10 seconds
        if current_time - last_check_time >= emotion_check_interval:
            try:
                # Analyze emotion from frame
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion'].lower()

                # If new emotion and it's mapped, open Tamil playlist in browser
                if emotion in emotion_playlists and emotion != last_emotion:
                    playlist_url = emotion_playlists[emotion]
                    webbrowser.open(playlist_url)
                    print(f"[{time.strftime('%H:%M:%S')}] Detected Emotion: {emotion} — Opened Tamil playlist.")
                    last_emotion = emotion

                last_check_time = current_time

            except Exception as e:
                print(f"Emotion detection error: {e}")

        # Show webcam feed
        cv2.imshow("Emotion Detector", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exited emotion detection.")
