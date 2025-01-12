import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

def generate_gradcam(model, image, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image.
    """
    grad_model = Model(
        inputs=[model.input],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_gradcam(image, heatmap, alpha=0.6):
    """
    Overlay Grad-CAM heatmap on the original image.
    """
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.image.resize(tf.expand_dims(heatmap, axis=-1), image.shape[:2])
    heatmap = tf.squeeze(heatmap) / 255.0

    heatmap_rgb = plt.cm.jet(heatmap.numpy())[:, :, :3]
    overlay_image = heatmap_rgb * alpha + image
    return np.clip(overlay_image, 0, 1)

def save_gradcam(image, heatmap, output_file, alpha=0.6):
    """
    Save Grad-CAM overlay as an image file.
    Args:
        image: Original image (H x W x 3)
        heatmap: Grad-CAM heatmap (H x W)
        output_file: Path to save the Grad-CAM image
        alpha: Opacity of the heatmap overlay
    """
    # Resize the heatmap to match the image size
    heatmap_resized = tf.image.resize(tf.expand_dims(heatmap, axis=-1), image.shape[:2])
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()

    # Convert the heatmap to RGB
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]

    # Overlay the heatmap on the original image
    overlay_image = heatmap_colored * alpha + image
    overlay_image = np.clip(overlay_image, 0, 1)  # Ensure pixel values are valid

    # Save the overlayed image
    plt.imsave(output_file, overlay_image)