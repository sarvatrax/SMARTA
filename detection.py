import cv2
from deepface import DeepFace
from config import CONFIG
import os
import json

class FaceDetection:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(CONFIG["haar_cascade"])

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 5)

    def predict_emotion(self, face_img):
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            return analysis[0]['dominant_emotion']
        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return "Unknown"

    def predict_age(self, face_img, user_name=None):
        try:
            analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
            age = int(analysis[0]['age'])
            
            # If user_name is provided, store the age prediction
            if user_name:
                self.store_age_prediction(user_name, age)
                
            return age
        except Exception as e:
            print(f"Age prediction error: {e}")
            return "Unknown"
    
    def store_age_prediction(self, user_name, age):
        """Store age prediction for a user"""
        
        # Ensure directory exists
        os.makedirs(CONFIG["age_dir"], exist_ok=True)
        
        # Path to store age data
        age_file = os.path.join(CONFIG["age_dir"], f"{user_name}.json")
        
        # Get existing age data or create new
        if os.path.exists(age_file):
            with open(age_file, 'r') as f:
                age_data = json.load(f)
        else:
            age_data = {"predictions": []}
        
        # Add new prediction
        age_data["predictions"].append(age)
        
        # Calculate median if we have enough predictions
        if len(age_data["predictions"]) >= 5:
            sorted_ages = sorted(age_data["predictions"])
            median_age = sorted_ages[len(sorted_ages) // 2]
            age_data["median_age"] = median_age
        
        # Save data
        with open(age_file, 'w') as f:
            json.dump(age_data, f)