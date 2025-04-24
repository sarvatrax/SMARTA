import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from detection import FaceDetection
from recognition import FaceRecognition
from table import UserTable
from config import CONFIG
import os
import csv
from datetime import datetime

class FaceRecognitionUI:
    def __init__(self):
        self.detector = FaceDetection()
        self.recognizer = FaceRecognition()
        self.video_cap = cv2.VideoCapture(0)
        self.root = tk.Tk()
        self.root.withdraw()
        self.initialize_attendance_file()
    
    def show_login_popup(self):
        try:
            popup = tk.Toplevel(self.root)
            popup.title("Face Recognition System")
            popup.state('zoomed')

            main_frame = tk.Frame(popup)
            main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

            tk.Label(main_frame, text="Welcome to\nFace Recognition System",
                    font=("Arial", 14), width=20, height=3).pack(pady=20)

            button_frame = tk.Frame(main_frame)
            button_frame.pack(pady=10)

            tk.Button(button_frame, text="Register", command=lambda: self.on_register(popup),
                     bg="green", fg="white", font=("Arial", 14), width=20, height=2).pack(side=tk.LEFT, padx=10)
            tk.Button(button_frame, text="Login", command=lambda: self.on_login(popup),
                     bg="blue", fg="white", font=("Arial", 14), width=20, height=2).pack(side=tk.LEFT, padx=10)

            table_container = tk.Frame(main_frame)
            table_container.pack(fill=tk.BOTH, expand=True, pady=10)

            # Fixed: Pass both detector and recognizer to UserTable
            self.user_table = UserTable(table_container, self.detector, self.recognizer)
            popup.mainloop()
            
        except tk.TclError as e:
            print(f"Error showing login popup: {e}")
            self.root = tk.Tk()
            self.root.withdraw()
            self.show_login_popup()
            
    def initialize_attendance_file(self):
        """Ensure attendance file exists with headers"""
        self.attendance_path = os.path.join(os.path.dirname(__file__), "attendance.csv")
        if not os.path.exists(self.attendance_path):
            with open(self.attendance_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Roll No", "Date", "Time", "Status"])

    def _log_attendance(self, recognized_name):
      try:
        if recognized_name == "Unknown":
            return False
            
        roll_no = self.recognizer.get_roll_no(recognized_name)
        
        with open(self.attendance_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                recognized_name.strip(),
                roll_no.strip(),
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"),
                "Present"
            ])
        return True
      except Exception as e:
        print(f"Attendance logging error: {e}")
        return False        

    def on_register(self, popup):
        name = simpledialog.askstring("Register", "Enter your name:")
        if not name:
            return
        roll_no = simpledialog.askstring("Register", "Enter your Roll No:")
        if not roll_no:
            return

        popup.destroy()
        self.register_new_user(name, roll_no)

    def on_login(self, popup):
        popup.destroy()
        self.show_camera()

    def register_new_user(self, name, roll_no):
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            faces = self.detector.detect_faces(frame)
            if len(faces) == 0:
                cv2.putText(frame, "Align face and press SPACE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(frame, "Press SPACE to capture", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.imshow("Register Face", frame)

            key = cv2.waitKey(1)
            if key == ord(' '):
                if len(faces) > 0:
                    face_img = frame[y:y+h, x:x+w]
                    # Remove redundant save - recognizer.register_user already saves the image
                    self.recognizer.register_user(name, roll_no, face_img)
                    messagebox.showinfo("Success", f"{name} registered successfully!")
                    break
            elif key == ord('E') or cv2.getWindowProperty("Register Face", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        self.show_login_popup()

    def show_camera(self):
        attendance_logged = False  # Track if we need to refresh the table
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            faces = self.detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                current_name = self.recognizer.recognize_face(face_img) or "Unknown"

                if current_name != "Unknown":
                     # Log attendance and set flag if successful
                 if self._log_attendance(current_name):
                    attendance_logged = True
                    
                    # Check if detector has these methods before calling them
                    age_text = ""
                    emotion_text = ""
                    
                    # Only call if these methods exist in your detector class
                    if hasattr(self.detector, 'predict_age'):
                        age = self.detector.predict_age(face_img, current_name)
                        age_text = f", {age}"
                    
                    if hasattr(self.detector, 'predict_emotion'):
                        emotion = self.detector.predict_emotion(face_img)
                        emotion_text = f", {emotion}" 
                    
                    display_text = f"{current_name}{age_text}{emotion_text}"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
                    cv2.putText(frame, display_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            
            # Refresh table periodically if attendance was logged
            if attendance_logged and hasattr(self, 'user_table'):
             self.user_table.refresh_table()
             attendance_logged = False  # Reset flag after refresh

            if cv2.waitKey(1) == ord('E') or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        if hasattr(self, 'user_table') and hasattr(self.user_table, 'content_frame') and self.user_table.content_frame.winfo_exists():
            self.user_table.refresh_table()
        self.show_login_popup()

    def cleanup(self):
        self.video_cap.release()
        cv2.destroyAllWindows()
        if self.root:
            self.root.destroy()

    def run(self):
        self.show_login_popup()
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionUI()
    try:
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        app.cleanup()