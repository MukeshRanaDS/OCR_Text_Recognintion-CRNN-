import configparser
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ImageProcessor import ImageProcessing
import pandas as pd

class DataPreparation:
    def __init__(self):
        pass

    def encode_to_labels(self, txt):
        dig_lst = []
        for index, char in enumerate(txt):
            try:
                dig_lst.append(char_list.index(char))
            except:
                print(char)
        return dig_lst

    def split_data(self, images, image_names, image_labels):
        total_size = len(images)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        images_train = np.array(images[:train_size])
        images_val = np.array(images[train_size:])

        image_labels_encoded_train = [self.encode_to_labels(label) for label in image_labels[:train_size]]
        image_labels_encoded_val = [self.encode_to_labels(label) for label in image_labels[train_size:]]

        train_input_length_train = np.array([31] * len(images_train))
        train_input_length_val = np.array([31] * len(images_val))

        image_labels_length_train = np.array([len(label) for label in image_labels[:train_size]])
        image_labels_length_val = np.array([len(label) for label in image_labels[train_size:]])

        max_label_len = 31
        print("max_label_len : ", max_label_len)

        # Create dataframes for train and val sets
        train_df = pd.DataFrame({"Image_name": image_names[:train_size], "Image_label": image_labels[:train_size]})
        val_df = pd.DataFrame({"Image_name": image_names[train_size:], "Image_label": image_labels[train_size:]})

        # Save dataframes to Excel files
        train_df.to_excel("train_name_label.xlsx", index=False)
        val_df.to_excel("val_name_label.xlsx", index=False)


        return {
            "Image_name_train": image_names[:train_size],
            "training_img_train": images_train,
            "train_padded_txt_train": pad_sequences(image_labels_encoded_train, maxlen=max_label_len, padding='post', value=len(char_list)),
            "train_label_length_train": image_labels_length_train,
            "train_input_length_train": train_input_length_train
        }, {
            "Image_name_val": image_names[train_size:],
            "training_img_val": images_val,
            "train_padded_txt_val": pad_sequences(image_labels_encoded_val, maxlen=max_label_len, padding='post', value=len(char_list)),
            "train_label_length_val": image_labels_length_val,
            "train_input_length_val": train_input_length_val
        }


# Load your images and other data
# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Read image_folder and num_images from config.ini
image_folder = config.get('ImageProcessing', 'image_folder')
num_images = int(config.get('ImageProcessing', 'num_images'))
char_list = config.get('ImageProcessing', 'char_list').split(',')
target_size_str = config.get('ImageProcessing', 'target_size')
target_size = tuple(map(int, target_size_str.split(',')))

processor = ImageProcessing(image_folder)
selected_images, image_names, image_labels, images = processor.extract_images(num_images, target_size=target_size)

# Instantiate DataPreparation class
data_prep = DataPreparation()

# Calculate maximum label length from the entire dataset
max_label_len = max(len(label) for label in image_labels)

# Split the data
train_data, val_data = data_prep.split_data(images, image_names, image_labels, )

# Unpack the split data
Image_name_train = train_data["Image_name_train"]
training_img_train = train_data["training_img_train"]
train_padded_txt_train = train_data["train_padded_txt_train"]
train_label_length_train = train_data["train_label_length_train"]
train_input_length_train = train_data["train_input_length_train"]

Image_name_val = val_data["Image_name_val"]
training_img_val = val_data["training_img_val"]
train_padded_txt_val = val_data["train_padded_txt_val"]
train_label_length_val = val_data["train_label_length_val"]
train_input_length_val = val_data["train_input_length_val"]

# Print each returned element for training dataset
print("Training Dataset:")
print("Image Names:", Image_name_train)
print("Training Images:", training_img_train)
print("Padded Text for Training:", train_padded_txt_train)
print("Label Length for Training:", train_label_length_train)
print("Input Length for Training:", train_input_length_train)
print("\n")
# Print each returned element for validation dataset
print("Validation Dataset:")
print("Image Names:", Image_name_val)
print("Training Images:", training_img_val)
print("Padded Text for Validation:", train_padded_txt_val)
print("Label Length for Validation:", train_label_length_val)
print("Input Length for Validation:", train_input_length_val)

# Print shape of train and val images
print("Shape of Train Images:", training_img_train.shape)
print("Shape of Validation Images:", training_img_val.shape)