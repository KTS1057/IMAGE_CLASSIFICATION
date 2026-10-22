"""
=============================================================================
Custom Multi-Class Image Classification Training Pipeline
Supports Animal, Fruit, or Custom Image Datasets via Transfer Learning
Author: Kratarth Srivastava (https://github.com/KTS1057)
=============================================================================
Usage:
    python train_custom_dataset.py --data_dir ./dataset --epochs 15 --batch_size 32
Dataset Directory Structure:
    dataset/
    ├── class_1 (e.g., Apple / Lion)
    ├── class_2 (e.g., Banana / Tiger)
    ├── class_3 (e.g., Orange / Bear)
    └── ...
=============================================================================
"""

import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2

def build_transfer_learning_model(num_classes, img_size=(224, 224)):
    """
    Constructs a modern transfer learning architecture using MobileNetV2
    with frozen base feature extractor and custom classification head.
    """
    # 1. Input layer
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))

    # 2. Data Augmentation Pipeline
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    x = data_augmentation(inputs)

    # 3. MobileNetV2 Preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 4. Pre-trained Base Model (Frozen)
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    # 5. Global Pooling & Dense Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="FineTuned_Vision_Classifier")
    return model, base_model

def train(data_dir, epochs=15, batch_size=32, img_size=(224, 224), output_model="fine_tuned_model.keras"):
    print("=" * 65)
    print("🚀 Initializing Custom Dataset Training Pipeline...")
    print(f"📁 Dataset Directory: {data_dir}")
    print(f"⚙️ Target Image Size: {img_size}")
    print(f"📦 Batch Size: {batch_size} | Epochs: {epochs}")
    print("=" * 65)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    # Load Training Dataset (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Load Validation Dataset (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\n✅ Detected {num_classes} Classes: {class_names}\n")

    # Save class mapping to JSON
    with open("class_labels.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print("💾 Saved class labels to class_labels.json")

    # Performance optimizations
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Build Model
    model, base_model = build_transfer_learning_model(num_classes, img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Training Callbacks
    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(output_model, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    print("\n⚡ Phase 1: Training Classification Head (Base Frozen)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks
    )

    print("\n⚡ Phase 2: Fine-Tuning Top Layers of Base Backbone...")
    base_model.trainable = True
    # Freeze the first 100 layers and unfreeze the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_epochs = 5
    total_epochs = epochs + fine_tune_epochs

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        callbacks=training_callbacks
    )

    print("\n" + "=" * 65)
    print(f"🎉 Training Complete! Best model saved to: {output_model}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Class Animal / Fruit Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=12, help="Number of initial training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_model", type=str, default="custom_classifier.keras", help="Target output model path")
    args = parser.parse_args()

    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, output_model=args.output_model)
