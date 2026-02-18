import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. SETTINGS & CONFIGURATION ---
# Path to your dataset folder (CHANGE THIS to your actual path)
DIRECTORY = "/content/DATASET"
CATEGORIES = ["with_mask", "without_mask"]


INIT_LR = 1e-4  # Learning Rate (how fast the AI learns)
EPOCHS = 10     # How many times to pass through the whole dataset
BS = 32         # Batch size (process 32 images at a time)

print("[INFO] Loading and preprocessing images...")

# --- 2. DATA LOADING & AUGMENTATION ---
# This tool loads images and creates "fake" variations (zooms, rotations)
# to make the model smarter.
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=preprocess_input, # Vital for MobileNetV2
    validation_split=0.2 # Use 20% of data for testing automatically
)

# Load Training Data (80%)
train_generator = train_datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset="training"
)

# Load Validation Data (20%)
val_generator = train_datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset="validation"
)

# --- 3. BUILD THE MODEL (MobileNetV2) ---
print("[INFO] Building the model...")

# Load the base MobileNetV2 model (pre-trained on ImageNet)
# include_top=False removes the head so we can add our own
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))
