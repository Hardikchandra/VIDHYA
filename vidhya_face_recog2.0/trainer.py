# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import os

# # # Define constants
# # IMG_SIZE = 224
# # BATCH_SIZE = 32
# # EPOCHS = 10
# # DATASET_PATH = "images"  # Replace with your dataset path
# # MODEL_PATH = "face_recognition_model.tflite"

# # # 1. Prepare Dataset
# # def prepare_dataset():
# #     datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# #     train_data = datagen.flow_from_directory(
# #         DATASET_PATH,
# #         target_size=(IMG_SIZE, IMG_SIZE),
# #         batch_size=BATCH_SIZE,
# #         class_mode='categorical',
# #         subset='training'
# #     )

# #     val_data = datagen.flow_from_directory(
# #         DATASET_PATH,
# #         target_size=(IMG_SIZE, IMG_SIZE),
# #         batch_size=BATCH_SIZE,
# #         class_mode='categorical',
# #         subset='validation'
# #     )
# #     return train_data, val_data
# # Define constants
# IMG_SIZE = 224
# BATCH_SIZE = 32
# EPOCHS = 10
# DATASET_PATH = "images"  # Root folder containing subfolders of person images
# MODEL_PATH = "face_recognition_model.tflite"

# # 1. Prepare Dataset
# def prepare_dataset():
#     datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

#     train_data = datagen.flow_from_directory(
#         DATASET_PATH,
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='training'
#     )

#     val_data = datagen.flow_from_directory(
#         DATASET_PATH,
#         target_size=(IMG_SIZE, IMG_SIZE),
#         batch_size=BATCH_SIZE,
#         class_mode='categorical',
#         subset='validation'
#     )

#     # Print class mappings
#     print("Class Indices (Person Names to Class Mapping):")
#     print(train_data.class_indices)
    
#     return train_data, val_data


# # 2. Build the Model
# def build_model(num_classes):
#     base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
#     base_model.trainable = False  # Freeze the base model

#     model = Sequential([
#         base_model,
#         GlobalAveragePooling2D(),
#         Dense(128, activation='relu'),
#         Dense(num_classes, activation='softmax')
#     ])
#     return model

# # 3. Train the Model
# def train_model():
#     train_data, val_data = prepare_dataset()
#     num_classes = train_data.num_classes

#     model = build_model(num_classes)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

#     # Save the trained model
#     model.save("face_recognition_model.h5")
#     print("Model trained and saved as face_recognition_model.h5")

# # 4. Convert to TFLite
# def convert_to_tflite():
#     model = tf.keras.models.load_model("face_recognition_model.h5")

#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()

#     with open(MODEL_PATH, 'wb') as f:
#         f.write(tflite_model)
#     print(f"Model converted to TFLite and saved as {MODEL_PATH}")

# # 5. Inference with TFLite
# def recognize_face(image_path):
#     interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Preprocess the image
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Perform inference
#     interpreter.set_tensor(input_details[0]['index'], img_array)
#     interpreter.invoke()

#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     predicted_class = np.argmax(output_data)
#     confidence = np.max(output_data)

#     print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

# # Train and convert the model
# train_model()
# convert_to_tflite()

# # Example inference
# # recognize_face("path_to_test_image.jpg")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json  # Ensure this is imported
from PIL import Image  # Pillow for image handling

# Check for PIL/Pillow
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is not installed. Please install it using `pip install Pillow`.")

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "images"  # Replace with your dataset path
MODEL_PATH = "face_recognition_model.tflite"

# 1. Prepare Dataset
def prepare_dataset():
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    return train_data, val_data

# 2. Build the Model
def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# 3. Train the Model
def train_model():
    train_data, val_data = prepare_dataset()
    save_class_indices(train_data)
    num_classes = train_data.num_classes
    # train_data, val_data = prepare_dataset()
    


    model = build_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    # Save the trained model
    model.save("face_recognition_model.h5")
    print("Model trained and saved as face_recognition_model.h5")

# 4. Convert to TFLite
def convert_to_tflite():
    model = tf.keras.models.load_model("face_recognition_model.h5")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to TFLite and saved as {MODEL_PATH}")

# 5. Inference with TFLite
def recognize_face(image_path):
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)

    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

# Save class indices to a JSON file
def save_class_indices(train_data):
    class_indices = train_data.class_indices
    with open("class_indices.json", "w") as f:
        json.dump(class_indices, f)
    print("Class indices saved to class_indices.json")


# Train and convert the model
train_model()
convert_to_tflite()


# Example inference
# recognize_face("path_to_test_image.jpg")
