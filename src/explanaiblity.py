# explainability.py
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_gradcam_heatmap(model, image, last_conv_layer_name="top_conv", pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    :param model: Trained Keras model.
    :param image: Preprocessed image (numpy array) with shape (height, width, channels).
    :param last_conv_layer_name: Name of the last convolutional layer in the model.
    :param pred_index: Index of the class for which Grad-CAM is computed. If None, uses predicted class.
    :return: Grad-CAM heatmap as a numpy array.
    """
    # Create a model mapping the input image to the activations of the last conv layer and predictions.
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Compute the gradients of the target class with respect to the feature map
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU to only keep positive contributions
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize the heatmap to a range of [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap /= max_val
    return heatmap.numpy()

def save_and_display_gradcam(image, heatmap, alpha=0.4, output_path="gradcam_output.jpg"):
    """
    Superimpose the Grad-CAM heatmap on the original image and save/display the result.
    :param image: Original image in BGR format.
    :param heatmap: Generated heatmap (values between 0 and 1).
    :param alpha: Transparency factor for heatmap overlay.
    :param output_path: File path to save the superimposed image.
    """
    # Rescale heatmap to 0-255 and resize to match the image dimensions
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap onto the original image
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    # Save the resulting image
    cv2.imwrite(output_path, superimposed_img)
    
    # Display the image
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()
