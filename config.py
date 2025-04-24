import os
import cv2

CONFIG = {
    "images_dir": os.path.join(os.path.expanduser("~"), "Desktop", "Face_Recog", "Face", "images"),
    "age_dir": os.path.join(os.path.expanduser("~"), "Desktop", "Face_Recog", "Face", "ages"),  # Add this line
    "haar_cascade": cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "recognition_threshold": 0.6,
    "model_name": "Facenet"
}