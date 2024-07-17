import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import requests

training_images_dir = "Training_images"
known_face_encodings = []
known_face_names = []

def perform_training():
    # Load training images and encode faces
    for filename in os.listdir(training_images_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(os.getcwd(),training_images_dir, filename)

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
    print('Training Complete!')   

def perform_inference(image):
    try:
        image_np = np.frombuffer(image, np.uint8)
        group_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        rgb_group_image = cv2.cvtColor(group_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_group_image)
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance_data = []
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_encoding = face_recognition.face_encodings(
                rgb_group_image, [face_locations[i]])[0]

            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.6
            )
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )

            best_match_index = None
            if True in matches:
                best_match_index = face_distances.argmin()

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            name = name + "@dsce.edu.in"

            attendance_data.append({"Name": name})
        return attendance_data
    except Exception as e:
        return None