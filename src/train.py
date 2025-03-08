# train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.data_preprocessing import load_and_preprocess_data, get_train_datagen
from src.model import build_model
from src.utils import configure_logging

# Configure logging
logger = configure_logging()

def train_model():
    try:
        logger.info("Loading and preprocessing data...")
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(target_size=(224, 224))
        
        # Create a validation split from training data (e.g., 10% validation)
        val_split = 0.1
        num_val = int(len(x_train) * val_split)
        x_val, y_val = x_train[:num_val], y_train[:num_val]
        x_train_new, y_train_new = x_train[num_val:], y_train[num_val:]
        
        # Data augmentation for training data
        train_datagen = get_train_datagen()
        train_generator = train_datagen.flow(x_train_new, y_train_new, batch_size=32)
        
        # Use a simple generator for validation data (no augmentation)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        val_generator = val_datagen.flow(x_val, y_val, batch_size=32)
        
        logger.info("Building the model...")
        model = build_model(num_classes=10, input_shape=(224, 224, 3))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Callbacks: EarlyStopping, LR reduction, and model checkpointing
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint('model/best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]
        
        logger.info("Starting training...")
        history = model.fit(train_generator,
                            epochs=30,
                            validation_data=val_generator,
                            callbacks=callbacks)
        
        # Ensure the model directory exists
        os.makedirs('model', exist_ok=True)
        # Save the final model
        model.save('model/final_model.h5')
        logger.info("Training complete. Model saved to 'model/final_model.h5'.")
    except Exception as e:
        logger.error("Error during training: %s", str(e))
        raise

if __name__ == "__main__":
    train_model()
