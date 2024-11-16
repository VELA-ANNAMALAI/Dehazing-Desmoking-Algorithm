from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

# Define the calculate_ssim function
def calculate_ssim(y_true, y_pred):
    y_true = tf.image.rgb_to_grayscale(y_true)
    y_pred = tf.image.rgb_to_grayscale(y_pred)
    ssim_index = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return ssim_index

# Define the calculate_psnr function
def calculate_psnr(y_true, y_pred):
    psnr_value = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return tf.reduce_mean(psnr_value)

# Optional: Tolerance-based accuracy function for pixel-wise accuracy
def calculate_accuracy(y_true, y_pred, epsilon=0.05):
    return tf.reduce_mean(tf.cast(tf.abs(y_true - y_pred) < epsilon, tf.float32))

# Custom objects dictionary
custom_objects = {
    'calculate_ssim': calculate_ssim,
    'calculate_psnr': calculate_psnr,
    'calculate_accuracy': calculate_accuracy
}

# Load the model with custom objects
with custom_object_scope(custom_objects):
    model = load_model('model.h5')

# Recompile the model with SSIM, PSNR, and MSE metrics
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[calculate_ssim, calculate_psnr])

# Load sample image
def load_sample_image(image_path, image_size):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Calculate SSIM, PSNR, MSE metrics
def calculate_metrics(haze, dehazed):
    haze_gray = tf.image.rgb_to_grayscale(haze)
    dehazed_gray = tf.image.rgb_to_grayscale(dehazed)
    
    # Calculate SSIM
    ssim_index = ssim(haze_gray.numpy().squeeze(), dehazed_gray.numpy().squeeze(), data_range=1.0)
    
    # Calculate PSNR
    psnr_value = tf.image.psnr(haze, dehazed, max_val=1.0).numpy()
    
    # Calculate MSE
    mse_value = np.mean((haze - dehazed) ** 2)
    
    return ssim_index, psnr_value, mse_value

# Plot only the haze and dehaze images
def plot_comparison(haze, dehazed):
    plt.figure(figsize=(12, 6))

    # Display the original haze image
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(haze, 0, 1))
    plt.title('Original Haze Image')
    plt.axis('off')

    # Display the dehazed image
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(dehazed, 0, 1))
    plt.title('Dehazed Image')
    plt.axis('off')

    plt.show()

# Load and evaluate sample image
sample_image_path = r"E:\project2\image-dehaze\data\test\2.jpg"
image_size = (256, 256)  # Ensure this matches the model's expected input size
sample_image = load_sample_image(sample_image_path, image_size)

# Predict dehazed image
dehazed_image = model.predict(sample_image)[0]
haze_image = img_to_array(load_img(sample_image_path, target_size=image_size)) / 255.0

# Display comparison
plot_comparison(haze_image, dehazed_image)
