tranner

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer

# Paths
images_path = "images/"
encodings_path = "face_encodings.pkl"
onnx_model_path = "face_recognition_model.onnx"  # Path to save ONNX model
hailo_model_path = "face_recognition_model.hailo"  # Path to save Hailo model

# Prepare data dictionary
face_data = {}

# Loop through each folder in the images directory
for person_name in os.listdir(images_path):
    person_folder = os.path.join(images_path, person_name)
    
    # Ensure it's a directory
    if os.path.isdir(person_folder):
        # Loop through each image in the person's folder
        for filename in os.listdir(person_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(person_folder, filename)
                
                # Load image and find face encodings
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
                image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                # Load your face encoder model
                face_encoder = tf.keras.models.load_model("face_encoder_model.h5")  # Adjust path as necessary
                face_encoding = face_encoder.predict(image_array)

                if face_encoding is not None and face_encoding.size > 0:
                    face_data[person_name] = face_encoding[0]
                    print(f"Encoded {person_name} from {filename}")

# Save encodings to a file
with open(encodings_path, "wb") as f:
    pickle.dump(face_data, f)

print("Encodings saved to 'face_encodings.pkl'")

# Train a model using the collected encodings
def train_model():
    X = []
    y = []
    for name, encoding in face_data.items():
        X.append(encoding)
        y.append(name)

    normalizer = Normalizer()
    X_normalized = normalizer.fit_transform(X)

    X_normalized = np.array(X_normalized)
    y = np.array(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_normalized.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(set(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_normalized, y, epochs=10, batch_size=8)

    # Save the model as ONNX
    tf.saved_model.save(model, "temp_model")
    os.system(f"python -m tf2onnx.convert --saved-model temp_model --output {onnx_model_path}")

    # Convert ONNX model to Hailo format (you need Hailo's conversion tool)
    os.system(f"hailo_convert --input {onnx_model_path} --output {hailo_model_path}")

    print(f"Model saved to '{hailo_model_path}'")

if __name__ == "__main__":
    train_model()