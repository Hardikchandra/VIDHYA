# # import tensorflow as tf
# # import numpy as np
# # from PIL import Image
# # import json

# # # Define constants
# # IMG_SIZE = 224
# # MODEL_PATH = "face_recognition_model.tflite"
# # CLASS_INDICES_PATH = "class_indices.json"  # Ensure this file maps indices to person names

# # # Load the TFLite model
# # def load_model():
# #     interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# #     interpreter.allocate_tensors()
# #     return interpreter

# # # # Load class indices mapping
# # # def load_class_indices():
# # #     try:
# # #         with open(CLASS_INDICES_PATH, "r") as f:
# # #             class_indices = json.load(f)
# # #         return {int(k): v for k, v in class_indices.items()}
# # #     except FileNotFoundError:
# # #         raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}")

# # # Load class indices mapping
# # def load_class_indices():
# #     try:
# #         with open(CLASS_INDICES_PATH, "r") as f:
# #             class_indices = json.load(f)
# #         # Check if keys are already integers; convert if they are strings
# #         return {int(k): v for k, v in class_indices.items()} if all(k.isdigit() for k in class_indices.keys()) else class_indices
# #     except FileNotFoundError:
# #         raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}")


# # # # Preprocess the image
# # # def preprocess_image(image_path):
# # #     img = Image.open(image_path).convert("RGB")
# # #     img = img.resize((IMG_SIZE, IMG_SIZE))
# # #     img_array = np.array(img) / 255.0  # Normalize to [0, 1]
# # #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# # #     return img_array
# # # Preprocess the image
# # def preprocess_image(image_path):
# #     img = Image.open(image_path).convert("RGB")
# #     img = img.resize((IMG_SIZE, IMG_SIZE))
# #     img_array = np.array(img) / 255.0  # Normalize to [0, 1]
# #     img_array = img_array.astype(np.float32)  # Explicitly cast to FLOAT32
# #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
# #     return img_array


# # # Perform inference
# # def recognize_face(image_path, interpreter, class_indices):
# #     input_details = interpreter.get_input_details()
# #     output_details = interpreter.get_output_details()

# #     # Preprocess the input image
# #     img_array = preprocess_image(image_path)

# #     # Set the tensor for input and invoke the model
# #     interpreter.set_tensor(input_details[0]['index'], img_array)
# #     interpreter.invoke()

# #     # Get the output tensor
# #     output_data = interpreter.get_tensor(output_details[0]['index'])
# #     predicted_class_idx = np.argmax(output_data)
# #     confidence = np.max(output_data)

# #     # Map index to class name
# #     predicted_class_name = class_indices.get(predicted_class_idx, "Unknown")

# #     return predicted_class_name, confidence

# # if __name__ == "__main__":
# #     # Example usage
# #     IMAGE_PATH = "WIN_20241216_10_29_06_Pro.jpg"  # Replace with the path to your test image

# #     # Load model and class indices
# #     interpreter = load_model()
# #     class_indices = load_class_indices()

# #     # Perform face recognition
# #     predicted_class, confidence = recognize_face(IMAGE_PATH, interpreter, class_indices)

# #     print(f"Predicted Person: {predicted_class}, Confidence: {confidence:.2f}")

# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json

# # Define constants
# IMG_SIZE = 224
# MODEL_PATH = "face_recognition_model.tflite"
# CLASS_INDICES_PATH = "class_indices.json"  # Ensure this file maps indices to person names

# # Load the TFLite model
# def load_model():
#     interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
#     interpreter.allocate_tensors()
#     return interpreter

# # Load class indices mapping
# def load_class_indices():
#     try:
#         with open(CLASS_INDICES_PATH, "r") as f:
#             class_indices = json.load(f)
#         # Convert keys to integers if necessary
#         return {int(k): v for k, v in class_indices.items()} if all(k.isdigit() for k in class_indices.keys()) else class_indices
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}")

# # Preprocess the image
# def preprocess_image(image_path):
#     img = Image.open(image_path).convert("RGB")
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1] and ensure FLOAT32
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# # Perform inference
# def recognize_face(image_path, interpreter, class_indices):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Preprocess the input image
#     img_array = preprocess_image(image_path)

#     # Ensure the input tensor type matches the expected dtype
#     expected_dtype = input_details[0]['dtype']
#     if img_array.dtype != expected_dtype:
#         img_array = img_array.astype(expected_dtype)

#     # Set the tensor for input and invoke the model
#     interpreter.set_tensor(input_details[0]['index'], img_array)
#     interpreter.invoke()

#     # Get the output tensor
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     predicted_class_idx = np.argmax(output_data)
#     confidence = np.max(output_data)

#     # Map index to class name
#     predicted_class_name = class_indices.get(predicted_class_idx, "Unknown")

#     return predicted_class_name, confidence

# if __name__ == "__main__":
#     # Example usage
#     IMAGE_PATH = "Rahul (3).jpg"  # Replace with the path to your test image

#     # Load model and class indices
#     interpreter = load_model()
#     class_indices = load_class_indices()

#     # Perform face recognition
#     predicted_class, confidence = recognize_face(IMAGE_PATH, interpreter, class_indices)

#     print(f"Predicted Person: {predicted_class}, Confidence: {confidence:.2f}")


import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Define constants
IMG_SIZE = 224
MODEL_PATH = "face_recognition_model.tflite"
CLASS_INDICES_PATH = "class_indices.json"  # Ensure this file maps indices to person names

# Load the TFLite model
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model loaded successfully.")
    return interpreter

# # Load class indices mapping
# def load_class_indices():
#     try:
#         with open(CLASS_INDICES_PATH, "r") as f:
#             class_indices = json.load(f)
#         # Ensure keys are integers for proper mapping
#         class_indices = {int(k): v for k, v in class_indices.items()}
#         print("Class indices loaded successfully:", class_indices)
#         return class_indices
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}")
#     except json.JSONDecodeError:
#         raise ValueError(f"Error decoding JSON in {CLASS_INDICES_PATH}")

# def load_class_indices():
#     try:
#         with open(CLASS_INDICES_PATH, "r") as f:
#             class_indices = json.load(f)
#         # Check if values are integers as expected
#         if all(isinstance(v, int) for v in class_indices.values()):
#             print("Class indices loaded successfully:", class_indices)
#             print("class_indicesss: "+ str(class_indices))
#             return class_indices
#         else:
#             raise ValueError(f"Class indices file {CLASS_INDICES_PATH} has invalid format. Values must be integers.")
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}")
#     except json.JSONDecodeError:
#         raise ValueError(f"Error decoding JSON in {CLASS_INDICES_PATH}")


# Load class indices mapping
def load_class_indices():
    try:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        # Flip the mapping to use indices as keys
        class_indices = {v: k for k, v in class_indices.items()}
        print("Flipped Class Indices Mapping:", class_indices)
        return class_indices
    except FileNotFoundError:
        raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON in {CLASS_INDICES_PATH}")


# Preprocess the image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = img_array.astype(np.float32)  # Explicitly cast to FLOAT32
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        print("Image preprocessed successfully.")
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

# Perform inference
def recognize_face(image_path, interpreter, class_indices):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Debugging input details
    print("Model Input Details:", input_details[0])

    # Preprocess the input image
    img_array = preprocess_image(image_path)

    # Debugging preprocessed image
    print("Preprocessed Image Shape:", img_array.shape, "Dtype:", img_array.dtype)

    # Set the tensor for input and invoke the model
    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {e}")

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Debugging output details
    print("Raw Output Data:", output_data)

    predicted_class_idx = np.argmax(output_data)
    confidence = np.max(output_data)

    # Map index to class name
    predicted_class_name = class_indices.get(predicted_class_idx, "Unknown")

    # Debugging prediction
    print(f"Predicted Index: {predicted_class_idx}, Confidence: {confidence}")
    print(f"Mapped Class Name: {predicted_class_name}")

    return predicted_class_name, confidence

if __name__ == "__main__":
    # Example usage
    IMAGE_PATH = "WIN_20241216_10_29_06_Pro.jpg"  # Replace with the path to your test image

    try:
        # Load model and class indices
        interpreter = load_model()
        class_indices = load_class_indices()

        # Perform face recognition
        predicted_class, confidence = recognize_face(IMAGE_PATH, interpreter, class_indices)

        print(f"Predicted Person: {predicted_class}, Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error: {e}")
