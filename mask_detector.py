import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import files

# --- 1. SETTINGS & CONFIGURATION ---
DIRECTORY = "/content/DATASET" 
CATEGORIES = ["with_mask", "without_mask"]

INIT_LR = 1e-4  
EPOCHS = 20     # Increased because we have "Smart Callbacks" now
BS = 32         

if not os.path.exists(DIRECTORY):
    print(f"[ERROR] Directory {DIRECTORY} not found!")
else:
    # --- 2. DATA LOADING & AUGMENTATION ---
    # Added 'validation_split' to ensure we have a test set the AI hasn't seen
    datagen = ImageDataGenerator(
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

    train_generator = datagen.flow_from_directory(
        DIRECTORY, target_size=(224, 224), batch_size=BS,
        class_mode="categorical", subset="training"
    )

    val_generator = datagen.flow_from_directory(
        DIRECTORY, target_size=(224, 224), batch_size=BS,
        class_mode="categorical", subset="validation"
    )

    # --- 3. BUILD THE MODEL ---
    print("[INFO] Initializing MobileNetV2 Base...")
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel) 
    headModel = Dense(len(CATEGORIES), activation="softmax")(headModel) 

    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    # --- 4. NEW: SMART CALLBACKS ---
    # 1. Stops training if the model stops improving
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # 2. Reduces Learning Rate if the model gets "stuck" (makes it more precise)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # --- 5. COMPILE AND TRAIN ---
    print("[INFO] Compiling and showing Model Summary...")
    model.summary() # This prints the architecture for your Viva!
    
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Starting Training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BS,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr]
    )

    # --- 6. SAVE AND DOWNLOAD ---
    model.save("mask_detector.keras")
    print("[INFO] Model Saved. Downloading now...")
    files.download("mask_detector.keras")

    # --- 7. FINAL VISUALIZATION ---
    N = len(history.history["loss"])
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()