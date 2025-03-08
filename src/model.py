# model.py
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model

def build_model(num_classes=10, input_shape=(224, 224, 3)):
    """
    Build an image classification model using EfficientNetB0 as the base.
    :param num_classes: Number of classes in the dataset.
    :param input_shape: Shape of input images.
    :return: Compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    # Load EfficientNetB0 with pre-trained ImageNet weights; exclude top layers.
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Apply dropout for regularization.
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model
