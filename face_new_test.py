from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import json

# Constants
IMG_SIZE = 224
MODEL_PATH = "quantized_face_recognition_model.tflite"
CLASS_INDICES_PATH = "class_indices.json"
CONFIDENCE_THRESHOLD = 0.7

def load_model():
    """Load the TFLite model."""
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def load_class_indices():
    """Load class mapping."""
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

def detect_face(image):
    """Detect and crop the largest face in an image."""
    if not hasattr(detect_face, "face_cascade"):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detect_face.face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_face.face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    padding = int(w * 0.1)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    face = image[y:y+h, x:x+w]
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE))

def recognize_face_from_image(image_path, interpreter, class_indices):
    """Recognize a face from a single image."""
    try:
        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Unable to load image from {image_path}")

        # Detect and crop face
        face_img = detect_face(frame)
        if face_img is None:
            return "Unknown"

        # Preprocess the image
        processed_img = np.expand_dims(face_img, axis=0).astype(np.float32) / 255.0

        # Set up inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        confidence = np.max(output)
        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown"

        person_idx = np.argmax(output)
        return class_indices.get(person_idx, "Unknown")

    except Exception as e:
        print(f"Error in recognition: {e}")
        return "Unknown"

# Example usage
if __name__ == "__main__":
    interpreter = load_model()
    class_indices = load_class_indices()

    # Replace 'test_image.jpg' with the path to your image
    image_path = "/home/vidhya/Desktop/final_vidhya/captured_images/image_20241218_101418.jpg"
    result = recognize_face_from_image(image_path, interpreter, class_indices)
    print(f"Recognition result: {result}")
