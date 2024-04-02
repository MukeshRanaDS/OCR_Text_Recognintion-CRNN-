import configparser
from keras.layers import Dense, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from Image_Premodeling_Preparation import DataPreparation
from ImageProcessor import ImageProcessing
import os
import pandas as pd

class ImageRecognitionTrainer:
    def __init__(self, image_folder, config_file='config.ini'):
        self.image_folder = image_folder
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.char_list = self.config.get('ImageProcessing', 'char_list').split(',')
        self.vocab_size = len(self.char_list)
        self.batch_size = int(self.config.get('ImageProcessing', 'batch_size'))
        self.epochs = int(self.config.get('ImageProcessing', 'epochs'))
        self.model = None

    def load_data(self):
        processor = ImageProcessing(self.image_folder)
        selected_images, image_names, image_labels, images = processor.extract_images(num_images=10, target_size=(128, 32))
        return images, image_names, image_labels

    def preprocess_data(self, images, image_names, image_labels):
        data_prep = DataPreparation()
        train_data, val_data = data_prep.split_data(images, image_names, image_labels)
        return train_data, val_data

    def build_model(self):
        inputs = Input(shape=(32, 128, 1))

        # convolution layer with kernel size (3,3)
        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        # pooling layer with kernel size (2,2)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
        # pooling layer with kernel size (2,1)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

        conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
        # Batch normalization layer
        batch_norm_5 = BatchNormalization()(conv_5)

        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

        conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

        squeezed = Lambda(lambda x: tf.squeeze(x, axis=1))(conv_7)

        # bidirectional LSTM layers with units=128
        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

        outputs = Dense(len(self.char_list) + 1, activation='softmax')(blstm_2)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    def train_model(self, training_data, validation_data, batch_size=32, epochs=10):
        training_img_train = training_data["training_img_train"]
        train_padded_txt_train = training_data["train_padded_txt_train"]
        training_img_val = validation_data["training_img_val"]
        train_padded_txt_val = validation_data["train_padded_txt_val"]

        # Create directory for saving epochs
        epoch_dir = "D:\office\epochs_50k"
        os.makedirs(epoch_dir, exist_ok=True)

        # Define checkpoint to save model after each epoch
        checkpoint = ModelCheckpoint(os.path.join(epoch_dir, "model_epoch_{epoch:02d}.h5"),
                                     monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

        # Train the model
        history = self.model.fit(
            x=training_img_train,
            y=to_categorical(train_padded_txt_train, num_classes=self.vocab_size + 1),
            validation_data=(training_img_val, to_categorical(train_padded_txt_val, num_classes=self.vocab_size + 1)),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[checkpoint]
        )

        # Save training history to CSV file
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("history_50k.csv", index=False)

        return history


if __name__ == "__main__":
    image_folder = r'C:\Datasets\MjSynth\90kDICT32px'
    trainer = ImageRecognitionTrainer(image_folder)
    images, image_names, image_labels = trainer.load_data()
    train_data, val_data = trainer.preprocess_data(images, image_names, image_labels)
    trainer.build_model()
    history = trainer.train_model(train_data, val_data)
