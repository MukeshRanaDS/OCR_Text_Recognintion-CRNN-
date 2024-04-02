import tensorflow as tf
import cv2
import configparser


# Function to preprocess input image
def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize image to match input shape of the model
    img = cv2.resize(img, (128, 32))
    # Normalize image
    img = img / 255.0
    # Add batch dimension and channel dimension
    img = img.reshape((1, 32, 128, 1))
    return img


# Function to decode predictions
def decode_predictions(preds, char_list):
    # Get index of highest probability for each timestep
    preds_index = tf.argmax(preds, axis=-1)
    # Convert index to characters
    decoded_text = ''.join([char_list[idx] for idx in preds_index[0] if idx != len(char_list)])
    return decoded_text


# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
# Read char_list from config.ini
char_list = config.get('ImageProcessing', 'char_list').split(',')

# Load the saved model
saved_model_path = r"D:\office\epochs_50k\model_epoch_500.h5"
loaded_model = tf.keras.models.load_model(saved_model_path)

# Path to input image for prediction
image_path = r"C:\Datasets\MjSynth\90kDICT32px\22\6\149_dickensian_21473.jpg"

# Preprocess input image
input_img = preprocess_image(image_path)

# Perform prediction
preds = loaded_model.predict(input_img)

# Decode predictions
predicted_text = decode_predictions(preds, char_list)

print("Predicted Text:", predicted_text)
