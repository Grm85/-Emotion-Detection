
import cv2              
from keras.models import model_from_json
import numpy as np

# Load the pre-trained emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the detected face image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to provide advisory messages based on detected emotions
def provide_advisory(emotion):
    advisories = {
        "happy": "Keep up the good work!",
        "sad": "Remember to take breaks and practice self-care.",
        "angry": "Take a deep breath and try to communicate calmly.",
        "neutral": "Stay focused and maintain a positive attitude.",
        "surprise": "Embrace new challenges with enthusiasm.",
        "fear": "It's okay to feel afraid. Take small steps to overcome your fears.",
        "disgust": "Focus on what you can control and let go of negativity.",
    }
    
    # Check if the detected emotion has a corresponding advisory message
    if emotion in advisories:
        return advisories[emotion]
    else:
        return "No specific advisory for this emotion."

# Capture video from webcam and perform real-time facial emotion detection
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    try: 
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_features = extract_features(face_img)
            
            # Predict emotion
            pred = model.predict(face_features)
            emotion_label = labels[pred.argmax()]
            
            # Display emotion label and advisory message
            cv2.putText(frame, f"Emotion: {emotion_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            advisory = provide_advisory(emotion_label)
            cv2.putText(frame, f"Advisory: {advisory}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Facial Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()

