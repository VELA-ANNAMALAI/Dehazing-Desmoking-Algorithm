import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Define paths
haze_dir = r'E:\project2\image-dehaze\data_set\haze'
dehaze_dir = r'E:\project2\image-dehaze\data_set\dehaze'

# Parameters
image_size = (256, 256)  # Resize images to this size
batch_size = 16
epochs = 50
learning_rate = 1e-4

# Data Generators
def create_data_generator(haze_dir, dehaze_dir, image_size, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)

    haze_generator = datagen.flow_from_directory(
        haze_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
        seed=42
    )

    dehaze_generator = datagen.flow_from_directory(
        dehaze_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
        seed=42
    )

    return haze_generator, dehaze_generator

haze_generator, dehaze_generator = create_data_generator(haze_dir, dehaze_dir, image_size, batch_size)

# Custom metrics
def calculate_ssim(y_true, y_pred):
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def calculate_psnr(y_true, y_pred):
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def calculate_accuracy(y_true, y_pred):
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    difference = tf.abs(y_true - y_pred)
    threshold = 0.05
    accurate_pixels = tf.less_equal(difference, threshold)
    accuracy = tf.reduce_mean(tf.cast(accurate_pixels, tf.float32))
    return accuracy

# Define Dehamer Model
def build_dehamer_model():
    model = Sequential([
        Input(shape=(image_size[0], image_size[1], 3)),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Conv2D(3, (3, 3), padding='same'),
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error', 
                  metrics=[tf.keras.metrics.MeanSquaredError(), calculate_ssim, calculate_psnr, calculate_accuracy])
    return model

# Define C2PNet Model
def build_c2pnet_model():
    model = Sequential([
        Input(shape=(image_size[0], image_size[1], 3)),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Conv2D(3, (3, 3), padding='same'),
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error', 
                  metrics=[tf.keras.metrics.MeanSquaredError(), calculate_ssim, calculate_psnr, calculate_accuracy])
    return model

# Define MITNet Model
def build_mitnet_model():
    model = Sequential([
        Input(shape=(image_size[0], image_size[1], 3)),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same'),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Conv2D(3, (3, 3), padding='same'),
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='mean_squared_error', 
                  metrics=[tf.keras.metrics.MeanSquaredError(), calculate_ssim, calculate_psnr, calculate_accuracy])
    return model

# Define Composite Model
def build_composite_model():
    # Build each model
    dehamer_model = build_dehamer_model()
    c2pnet_model = build_c2pnet_model()
    mitnet_model = build_mitnet_model()

    # Define input layer
    input_image = Input(shape=(image_size[0], image_size[1], 3))
    
    # Pass input through each model sequentially
    dehazed_output = dehamer_model(input_image)
    c2pnet_output = c2pnet_model(dehazed_output)
    final_output = mitnet_model(c2pnet_output)

    # Create the final composite model
    composite_model = Model(inputs=input_image, outputs=final_output)
    composite_model.compile(optimizer=Adam(learning_rate=learning_rate), 
                            loss='mean_squared_error', 
                            metrics=[tf.keras.metrics.MeanSquaredError(), calculate_ssim, calculate_psnr, calculate_accuracy])
    return composite_model

# Define the composite model
model = build_composite_model()

# Calculate steps per epoch
steps_per_epoch = min(len(haze_generator), len(dehaze_generator))

# Train Model
history = model.fit(
    zip(haze_generator, dehaze_generator),
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=1
)

# Save the model in both .keras and .h5 formats with dynamic filenames
model.save('model.keras')
model.save('model.h5')

# Plot training metrics
def plot_training_metrics(history):
    plt.figure(figsize=(18, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MSE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mean_squared_error'], label='Mean Squared Error', color='orange')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # Plot SSIM
    plt.subplot(2, 2, 3)
    plt.plot(history.history['calculate_ssim'], label='SSIM', color='green')
    plt.title('Structural Similarity Index (SSIM)')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()

    # Plot PSNR
    plt.subplot(2, 2, 4)
    plt.plot(history.history['calculate_psnr'], label='PSNR', color='purple')
    plt.title('Peak Signal-to-Noise Ratio (PSNR)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.legend()

    plt.show()

plot_training_metrics(history)