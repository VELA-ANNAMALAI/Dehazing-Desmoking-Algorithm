import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_from_directory(directory, target_size=None):
   
    images = []
    labels = []

    for class_label in os.listdir(directory):
        class_dir = os.path.join(directory, class_label)
        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(class_dir, filename)
                try:
                    image = Image.open(file_path).convert('RGB')
                    if target_size:
                        image = image.resize(target_size)
                    image = np.array(image) / 255.0  # Normalize to [0, 1]
                    images.append(image)
                    labels.append(int(class_label[-1]))  # Assumes labels are like 'class1' or 'class2'
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    return np.array(images), np.array(labels)

def preprocess_train_data(haze_dir, dehaze_dir, target_size=(256, 256)):
    
    print("Loading and preprocessing haze images...")
    haze_images, _ = load_images_from_directory(haze_dir, target_size=target_size)
    
    print("Loading and preprocessing dehaze images...")
    dehaze_images, _ = load_images_from_directory(dehaze_dir, target_size=target_size)

    return haze_images, dehaze_images

def load_test_data(haze_dir, dehaze_dir):
   
    print("Loading test haze images...")
    test_haze, _ = load_images_from_directory(haze_dir)
    
    print("Loading test dehaze images...")
    test_dehaze, _ = load_images_from_directory(dehaze_dir)

    return test_haze, test_dehaze

if __name__ == "__main__":
    haze_dir = r'E:\project2\image-dehaze\data_set\haze'  # Update with the path to your haze images directory
    dehaze_dir = r'E:\project2\image-dehaze\data_set\dehaze'  # Update with the path to your dehaze images directory

    # Load and preprocess training data
    train_haze, train_dehaze = preprocess_train_data(haze_dir, dehaze_dir)

    # Save preprocessed training data
    np.save('train_haze_preprocessed.npy', train_haze)
    np.save('train_dehaze_preprocessed.npy', train_dehaze)

    # Load test data (without preprocessing)
    test_haze, test_dehaze = load_test_data(haze_dir, dehaze_dir)

    # Save test data
    np.save('test_haze.npy', test_haze)
    np.save('test_dehaze.npy', test_dehaze)

    print("Data loading, preprocessing, and saving completed.")
