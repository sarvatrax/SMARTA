import os
import cv2
import json
from deepface import DeepFace
from config import CONFIG

class FaceRecognition:
    def __init__(self):
        self.roll_numbers_dir = os.path.join(os.path.dirname(__file__), "roll_numbers")
        os.makedirs(self.roll_numbers_dir, exist_ok=True)

    def register_user(self, name, roll_no, face_img):
        # Save image (name only)
        img_path = os.path.join(CONFIG["images_dir"], f"{name}.jpg")
        cv2.imwrite(img_path, face_img)
        
        # Save roll number separately
        roll_path = os.path.join(self.roll_numbers_dir, f"{name}.txt")
        with open(roll_path, "w") as f:
            f.write(roll_no)
        
        return True

    def recognize_face(self, face_img):
        try:
            matches = DeepFace.find(
                img_path=face_img,
                db_path=CONFIG["images_dir"],
                model_name=CONFIG["model_name"],
                enforce_detection=False,
                silent=True
            )

            if matches and not matches[0].empty:
                best_match = matches[0].iloc[0]
                if best_match["distance"] < CONFIG["recognition_threshold"]:
                    name = os.path.splitext(os.path.basename(best_match["identity"]))[0]
                    return name
            return "Unknown"
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown"

    def get_roll_no(self, name):
        roll_path = os.path.join(self.roll_numbers_dir, f"{name}.txt")
        if os.path.exists(roll_path):
            with open(roll_path, "r") as f:
                return f.read().strip()
        return "N/A"