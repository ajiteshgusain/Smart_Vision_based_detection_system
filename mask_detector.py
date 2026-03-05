import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import files # For easy downloading

# --- 1. SETTINGS & CONFIGURATION ---
DIRECTORY = "/content/DATASET" # Ensure this matches your unzipped folder name

# Check if directory exists to avoid errors
if not os.path.exists(DIRECTORY):
    print(f"[ERROR] Directory {DIRECTORY} not found! Please check your folder name.")
else:
    INIT_LR = 1e-4  
    EPOCHS = 15     # Increased slightly; EarlyStopping will handle it if it's too much
    BS = 32         

    print("[INFO] Loading and preprocessing images...")

    # --- 2. DATA LOADING & AUGMENTATION ---
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        preprocessing_function=preprocess_input, 
        validation_split=0.2 
    )

    train_generator = train_datagen.flow_from_directory(
        DIRECTORY,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="training"
    )

    val_generator = train_datagen.flow_from_directory(
        DIRECTORY,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="validation"
    )

    # --- 3. BUILD THE MODEL ---
    print("[INFO] Building the model...")
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel) 
    headModel = Dense(2, activation="softmax")(headModel) 

    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze base model layers
    for layer in baseModel.layers:
        layer.trainable = False

    # --- 4. COMPILE AND TRAIN ---
    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # ADDED: Early Stopping (Stops training if val_loss doesn't improve for 3 rounds)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("[INFO] Training head...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BS,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # --- 5. SAVE AND DOWNLOAD ---
    print("[INFO] Saving mask detector model...")
    model.save("mask_detector.keras")
    
    # This will trigger a browser download automatically!
    files.download("mask_detector.keras")

    # --- 6. PLOT RESULTS ---
    N = len(history.history["loss"]) # Handle cases where early stopping kicks in
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Results")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()
    print("[INFO] Process Complete.")