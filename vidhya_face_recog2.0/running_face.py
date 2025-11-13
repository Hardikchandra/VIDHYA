import tensorflow as tf
import numpy as np
import cv2
import json

# Define constants
IMG_SIZE = 224
MODEL_PATH = "quantized_face_recognition_model.tflite"
CLASS_INDICES_PATH = "class_indices.json"

def load_model():
    """Load the quantized TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def load_class_indices():
    """Load class indices mapping from the JSON file."""
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # Flip mapping

def preprocess_image(image_path):
    """Preprocess image for input."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

def recognize_face(image_path, interpreter, input_details, output_details, class_indices):
    """Perform face recognition on the image."""
    img_array = preprocess_image(image_path)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_idx = np.argmax(output_data)
    confidence = np.max(output_data)

    predicted_class_name = class_indices.get(predicted_class_idx, "Unknown")
    return predicted_class_name, confidence

if _name_ == "_main_":
    # Test Image Path (Change this to your image path)
    IMAGE_PATH = "karan (3).jpg"

    try:
        # Load model and class indices
        interpreter, input_details, output_details = load_model()
        class_indices = load_class_indices()

        # Perform face recognition
        predicted_class, confidence = recognize_face(
            IMAGE_PATH, interpreter, input_details, output_details, class_indices
        )

        # Display the result
        print(f"Predicted Person: {predicted_class}, Confidence: {confidence:.2f}")

    except Exception as e:
        print(f"Error: {e}")
