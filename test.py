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

# Create a dictionary of custom objects
custom_objects = {
    'calculate_ssim': calculate_ssim,
    'calculate_psnr': calculate_psnr,
    'calculate_accuracy': calculate_accuracy  # Include this only if you still want accuracy
}

# Load the model with custom objects and handle possible deserialization issues
with custom_object_scope(custom_objects):
    model = load_model('model.h5')

# Recompile the model focusing on SSIM, PSNR, and MSE
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[calculate_ssim, calculate_psnr])

# Function to load a sample image for evaluation
def load_sample_image(image_path, image_size):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Function to calculate SSIM, PSNR, and MSE
def calculate_metrics(original, dehazed):
    # Convert the images to grayscale for SSIM
    original_gray = tf.image.rgb_to_grayscale(original)
    dehazed_gray = tf.image.rgb_to_grayscale(dehazed)
    
    # Convert tensors to NumPy arrays for SSIM calculation
    ssim_index = ssim(original_gray.numpy().squeeze(), dehazed_gray.numpy().squeeze(), data_range=1.0)
    
    # Calculate PSNR
    psnr_value = tf.image.psnr(original, dehazed, max_val=1.0).numpy()
    
    # Calculate MSE
    mse_value = np.mean((original - dehazed) ** 2)
    
    return ssim_index, psnr_value, mse_value

# Function to plot the comparison between the original, dehazed, and ground truth images
def plot_comparison(original, dehazed, ground_truth, ssim_index, psnr_value, mse_value):
    plt.figure(figsize=(18, 6))

    # Display the original haze image
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(original, 0, 1))
    plt.title('Original Haze Image')
    plt.axis('off')

    # Display the dehazed image
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(dehazed, 0, 1))
    plt.title('Dehazed Image')
    plt.axis('off')

    # Display the ground truth image
    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(ground_truth, 0, 1))
    plt.title('Ground Truth Image')
    plt.axis('off')

    # Adjust the layout to create space for the text
    plt.subplots_adjust(bottom=0.5)

    # Display the SSIM, PSNR, and MSE below the images
    plt.figtext(0.5, 0.40, 
                f'SSIM: {ssim_index:.4f}, PSNR: {psnr_value:.2f} dB, MSE: {mse_value:.6f}', 
                ha='center', va='center', fontsize=12, wrap=True)

    plt.show()

# Example image path (update with your own image path)
sample_image_path = r"E:\project2\image-dehaze\data_set\haze\class1\481.jpg"
ground_truth_image_path = r"E:\project2\image-dehaze\data_set\dehaze\class1\481.jpg"
image_size = (256, 256)  # This should match the size your model was trained on

# Prepare the sample image
sample_image = load_sample_image(sample_image_path, image_size)

# Predict dehazed image
dehazed_image = model.predict(sample_image)[0]

# Load original and ground truth images for comparison
original_image = img_to_array(load_img(sample_image_path, target_size=image_size)) / 255.0
ground_truth_image = img_to_array(load_img(ground_truth_image_path, target_size=image_size)) / 255.0

# Calculate metrics (SSIM, PSNR, MSE)
ssim_index, psnr_value, mse_value = calculate_metrics(ground_truth_image, dehazed_image)

# Plot the results
plot_comparison(original_image, dehazed_image, ground_truth_image, ssim_index, psnr_value, mse_value)
