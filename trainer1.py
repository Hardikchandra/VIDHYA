import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder


# Paths
images_path = "images/"
encodings_path = "face_encodings.pkl"
onnx_model_path = "face_recognition_model.onnx"  # Path to save ONNX model
hailo_model_path = "face_recognition_model.hailo"  # Path to save Hailo model
face_encoder_model_path = "face_encoder_model.h5"  # Path to save the trained face encoder model

# Prepare data dictionary
face_data = {}

# Function to create and train a face encoder model
def create_face_encoder_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(160, 160, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu')  # Output layer for face encoding
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load images and generate face encodings
def load_images_and_encode():
    global face_data
    for person_name in os.listdir(images_path):
        person_folder = os.path.join(images_path, person_name)
        
        # Ensure it's a directory
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_folder, filename)
                    
                    # Load image and preprocess it
                    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
                    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)

                    # Predict face encoding
                    face_encoding = face_encoder.predict(image_array)

                    if face_encoding is not None and face_encoding.size > 0:
                        face_data[person_name] = face_encoding[0]
                        print(f"Encoded {person_name} from {filename}")

# Save encodings to a file
def save_encodings():
    with open(encodings_path, "wb") as f:
        pickle.dump(face_data, f)
    print("Encodings saved to 'face_encodings.pkl'")

def train_model():
    X = []
    y = []
    for name, encoding in face_data.items():
        X.append(encoding)
        y.append(name)

    normalizer = Normalizer()
    X_normalized = normalizer.fit_transform(X)

    X_normalized = np.array(X_normalized)

    # Convert string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Now y_encoded contains integer labels
    y_encoded = np.array(y_encoded)

    # Ensure the output layer matches the number of unique classes
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_normalized.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')  # Adjusted to number of unique classes
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_normalized, y_encoded, epochs=10, batch_size=8)

    # Save the model as ONNX
    tf.saved_model.save(model, "temp_model")
    os.system(f"python -m tf2onnx.convert --saved-model temp_model --output {onnx_model_path}")

    # Convert ONNX model to Hailo format (you need Hailo's conversion tool)
    os.system(f"hailo_convert --input {onnx_model_path} --output {hailo_model_path}")

    print(f"Model saved to '{hailo_model_path}'")

if __name__ == "__main__":
    # Create and train the face encoder model
    face_encoder = create_face_encoder_model()
    
    # Load images and generate face encodings
    load_images_and_encode()
    
    # Save the face encoder model
    face_encoder.save(face_encoder_model_path)
    print(f"Face encoder model saved to '{face_encoder_model_path}'")

    # Save encodings
    save_encodings()

    # Train a model using the collected encodings
    train_model()
