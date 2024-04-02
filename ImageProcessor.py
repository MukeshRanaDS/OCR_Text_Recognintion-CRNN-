import os
import fnmatch
import cv2
import configparser
import numpy as np
import pandas as pd

class ImageProcessing:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = []

    def populate_image_files(self, num_images):
        for root, dirnames, filenames in os.walk(self.image_folder):
            for f_name in fnmatch.filter(filenames, '*.jpg'):
                if len(self.image_files) < num_images:
                    self.image_files.append(os.path.join(root, f_name))
                else:
                    return

    def extract_images(self, num_images, target_size=None):
        if not self.image_files:
            self.populate_image_files(num_images)

        num_images = min(num_images, len(self.image_files))
        selected_images = self.image_files[:num_images]

        image_names = []
        image_labels = []
        images = []
        for file in selected_images:
            file_name = os.path.splitext(os.path.basename(file))[0]
            image_names.append(file_name)
            label = file_name.split('_')[1]
            image_labels.append(label)
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if target_size is not None:
                image = cv2.resize(image, target_size)
            else:
                # If target size is not provided, use the original image size
                target_size = (image.shape[1], image.shape[0])
            image = image[..., np.newaxis]  # Add channel dimension
            image = image / 255.0  # Normalize
            images.append(image)
            if target_size is not None:
                print(f"Resized Image Shape: {image.shape}")
            else:
                print(f"Original Image Shape: {image.shape}")
        print("Image Processor Done...")

        # Save image_names and image_labels to Excel file
        df = pd.DataFrame({"image_names": image_names, "image_labels": image_labels})
        df.to_excel("image_name_label.xlsx", index=False)

        return selected_images, image_names, image_labels, images


# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')
image_folder = config['ImageProcessing']['image_folder']
num_images = int(config['ImageProcessing']['num_images'])
target_size = (128, 32)

# Example usage:
processor = ImageProcessing(image_folder)
selected_images, image_names, image_labels, images = processor.extract_images(num_images, target_size)

print("Selected Images:", selected_images)
print("Image Names:", image_names)
print("Image Labels:", image_labels)
# print("Images:", images)
print(len(images[0]), len(images[0][1]), len(images[0][1][2]))