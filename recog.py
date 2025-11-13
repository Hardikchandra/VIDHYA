
import cv2
import numpy as np
import pickle
from hailo_platform import Model, HailoNPU
import face_recognition  # Ensure you have this library installed

# Load encodings
with open("face_encodings.pkl", "rb") as f:
    face_data = pickle.load(f)

# Separate encodings and names
known_face_encodings = list(face_data.values())
known_face_names = list(face_data.keys())

# Load the Hailo model
model = Model("face_recognition_model.hailo")
npu = HailoNPU()

def recognize_faces_in_video():
    video_capture = cv2.VideoCapture(0)  # For webcam
    frame_counter = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame_counter += 1
        if frame_counter % 5 == 0:
            recognized_names = recognize_faces(frame)

            if recognized_names:
                print("Recognized: ", ", ".join(recognized_names))
                return recognized_names[0]

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return None

def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        input_data = np.array(face_encoding).reshape(1, -1)
        output_data = npu.run_model(model, input_data)

        best_match_index = np.argmax(output_data[0])
        confidence = output_data[0][best_match_index]

        name = "Unknown"
        if confidence > 0.5:
            name = known_face_names[best_match_index]

        recognized_names.append(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return recognized_names

if __name__ == "__main__":
    recognize_faces_in_video()