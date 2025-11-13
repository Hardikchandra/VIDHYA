from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
import json
import os
import time
from datetime import datetime

# Constants
IMG_SIZE = 224
MODEL_PATH = "quantized_face_recognition_model.tflite"
CLASS_INDICES_PATH = "class_indices.json"

def capture_image():
    # Initialize the camera
    camera = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return None

    # Create a directory to save images if it doesn't exist
    image_dir = "captured_images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Create a window to display the camera feed
    cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)

    # Start capturing the video feed
    start_time = time.time()
    image_path = None  # Initialize image_path variable

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame in the window
        cv2.imshow("Camera Feed", frame)

        # Check if 5 seconds have passed
        if time.time() - start_time > 5:
            # Capture the image
            image_path = os.path.join(image_dir, f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved at: {image_path}")
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()

    # Return the path of the saved image
    return image_path
    
def recognize_face(image_path):
    """Perform face recognition on the image."""
    # Initialize global variables
    interpreter = None
    input_details = None
    output_details = None
    class_indices = None

    # Load model
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"Error loading the model: {e}")
        return "Error", 0.0

    # Load class indices
    try:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = {int(v): k for k, v in json.load(f).items()}
    except Exception as e:
        print(f"Error loading class indices: {e}")
        return "Error", 0.0

    # Preprocess image
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image from {image_path}. Check the file path.")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalize
        img_array = np.expand_dims(img.astype(np.float32), axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return "Error", 0.0

    # Perform face recognition
    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_idx = np.argmax(output_data)
        confidence = np.max(output_data)

        # Check if the predicted index is valid
        if predicted_class_idx in class_indices:
            predicted_class_name = class_indices[predicted_class_idx]
        else:
            predicted_class_name = "Unknown"

        return predicted_class_name
    except Exception as e:
        print(f"Error during inference: {e}")
        return "Error", 0.0

# If you want to test the function in the same file
if __name__ == "__main__":
    IMAGE_PATH = "/home/vidhya/Desktop/final_vidhya/WIN_20241216_10_29_06_Pro.jpg"

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image path {IMAGE_PATH} does not exist.")
    else:
        predicted_class = recognize_face(IMAGE_PATH)
        print(f"Predicted Person: {predicted_class}")
