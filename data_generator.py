import cv2
import numpy as np
import os

def load_and_preprocess_images(directory, image_size):
    """
    Load images from a directory, resize them, and normalize pixel values.

    :param directory: Directory containing subdirectories of images.
    :param image_size: Tuple (width, height) to resize images.
    :return: NumPy array of preprocessed images.
    """
    file_names = []
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            file_names.extend([os.path.join(class_path, fname) for fname in os.listdir(class_path) if fname.endswith(('.png', '.jpg', '.jpeg'))])
    
    images = [cv2.resize(cv2.imread(file_name), image_size) for file_name in file_names]
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    return images

def create_data_generator(haze_dir, dehaze_dir, image_size, batch_size):
    """
    Create a data generator that yields batches of haze and dehaze images.

    :param haze_dir: Directory containing haze images organized in subdirectories.
    :param dehaze_dir: Directory containing dehaze images organized in subdirectories.
    :param image_size: Tuple (width, height) to resize images.
    :param batch_size: Number of samples per batch.
    :return: A generator yielding batches of haze and dehaze images.
    """
    # Load and preprocess images
    haze_images = load_and_preprocess_images(haze_dir, image_size)
    dehaze_images = load_and_preprocess_images(dehaze_dir, image_size)

    num_samples = min(len(haze_images), len(dehaze_images))
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    haze_images = haze_images[indices]
    dehaze_images = dehaze_images[indices]

    def data_generator():
        """
        Generator function that yields batches of haze and dehaze images.
        """
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            yield haze_images[start:end], dehaze_images[start:end]

    return data_generator()

# Example usage
if __name__ == "__main__":
    # Define paths and parameters
    haze_dir = 'E:\\project2\\image-dehaze\\data\\haze'
    dehaze_dir = 'E:\\project2\\image-dehaze\\data\\dehaze'
    image_size = (256, 256)  # Resize images to this size
    batch_size = 16

    # Create data generator
    train_generator = create_data_generator(haze_dir, dehaze_dir, image_size, batch_size)

    # Example: Retrieve a batch of images
    batch_haze, batch_dehaze = next(train_generator)
    print(f"Batch haze shape: {batch_haze.shape}")
    print(f"Batch dehaze shape: {batch_dehaze.shape}")
