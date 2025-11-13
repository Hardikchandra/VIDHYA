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
    """Load the TFLite model"""
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def load_class_indices():
    """Load class mappings"""
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

def detect_face(image):
    """Lightweight face detection using Haar cascades"""
    if not hasattr(detect_face, "face_cascade"):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detect_face.face_cascade = cv2.CascadeClassifier(cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_face.face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
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

def recognize_face(frame, interpreter, class_indices):
    """Perform face recognition"""
    try:
        face_img = detect_face(frame)
        if face_img is None:
            return "Unknown"
        
        processed_img = np.expand_dims(face_img, axis=0).astype(np.float32) / 255.0
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        confidences = output[0]
        max_confidence = np.max(confidences)
        person_idx = np.argmax(confidences)
        person_name = class_indices.get(person_idx, "Unknown")
        
        if max_confidence < CONFIDENCE_THRESHOLD + 0.1:
            return "Unknown"
        return person_name
    except Exception as e:
        print(f"Recognition Error: {e}")
        return "Unknown"

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    interpreter = load_model()
    class_indices = load_class_indices()
    print("Model loaded and ready")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue
            
            result = recognize_face(frame, interpreter, class_indices)
            print(result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
