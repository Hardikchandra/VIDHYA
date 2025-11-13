import face_recognition
import pickle
import cv2
import numpy as np

# Load encodings
with open("face_encodings.pkl", "rb") as f:
    face_data = pickle.load(f)

# Separate encodings and names
known_face_encodings = list(face_data.values())
known_face_names = list(face_data.keys())

def recognize_faces_in_video():
    """
    Captures video from the webcam and recognizes faces in real-time.
    Prints recognized names and closes the window if a name is recognized.
    Returns the first recognized name or None if no faces are recognized.
    """
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)  # For webcam

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()

        # Recognize faces in the captured frame
        recognized_names = recognize_faces(frame)

        # Print recognized names
        if recognized_names:  # Check if any names were recognized
            print("Recognized: ", ", ".join(recognized_names))
            return recognized_names[0]  # Return the first recognized name

        # Display the result
        cv2.imshow("Face Recognition", frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return None  # Return None if no faces were recognized

def recognize_faces(frame):
    """
    Recognizes faces in the given frame and returns a list of recognized names.
    Also draws rectangles around recognized faces on the frame.

    Args:
        frame: The input image frame to process.

    Returns:
        List of recognized names.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        # Find the closest match
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        else:
            # Find the closest face match if no matches
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]

        recognized_names.append(name)

        # Draw a box around the face and label with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return recognized_names
