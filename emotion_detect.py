import cv2
from deepface import DeepFace
import pyttsx3
import time
import threading

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize webcam
cap = cv2.VideoCapture(0)

last_spoken_emotion = ""
last_spoken_time = 0

def speak_emotion(text):
    engine.say(text)
    engine.runAndWait()

frame_counter = 0
emotion = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze every 30 frames (~1 sec if ~30 FPS)
    frame_counter += 1
    if frame_counter % 30 == 0:
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            detected_emotion = results[0]['dominant_emotion']

            current_time = time.time()
            if detected_emotion != last_spoken_emotion or (current_time - last_spoken_time) > 5:
                threading.Thread(target=speak_emotion, args=(f"You look {detected_emotion}",)).start()
                last_spoken_emotion = detected_emotion
                last_spoken_time = current_time

            emotion = detected_emotion

        except Exception:
            emotion = "Undetected"

    # Display emotion on frame
    cv2.putText(frame, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition with Audio", frame)

    # WaitKey returns -1 if no key is pressed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
