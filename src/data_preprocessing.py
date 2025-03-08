# data_preprocessing.py
import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(target_size=(224, 224)):
    """
    Load CIFAR-10 dataset and preprocess images by resizing and normalizing.
    :param target_size: Desired image size (width, height)
    :return: Tuple containing preprocessed (x_train, y_train) and (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Resize images to target size
    x_train_resized = np.array([cv2.resize(img, target_size) for img in x_train])
    x_test_resized = np.array([cv2.resize(img, target_size) for img in x_test])

    # Normalize pixel values to [0,1]
    x_train_norm = x_train_resized.astype('float32') / 255.0
    x_test_norm = x_test_resized.astype('float32') / 255.0

    return (x_train_norm, y_train), (x_test_norm, y_test)

def get_train_datagen():
    """
    Create an ImageDataGenerator for training with data augmentation.
    :return: Configured ImageDataGenerator instance.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    return train_datagen
