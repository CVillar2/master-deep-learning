import numpy as np
import glob, os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def prepare_data_generator_VAE(data_folder, input_dim=(128, 128, 3), batch_size=512):
    if not os.path.exists(data_folder):
        raise ValueError("Invalid data folder path.")

    filenames = np.array(glob.glob(os.path.join(data_folder, '*/*.jpg')))
    num_images = len(filenames)
    print(f"Total number of images: {num_images}")

    data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(data_folder, 
        target_size = input_dim[:2],
        batch_size = batch_size,
        shuffle = True,
        class_mode = 'input',
        subset = 'training',
        color_mode=('grayscale' if input_dim[2]==1 else 'rgb')
        )

    return data_flow, num_images


def prepare_data_generator_GAN(data_folder, image_size, batch_size, color_mode):
    """
    Load and preprocess the data using TensorFlow's image_dataset_from_directory and preprocessing function.
    """

    # Load the data
    train_data = tf.keras.utils.image_dataset_from_directory(
        data_folder,
        labels=None,
        color_mode=color_mode,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        interpolation="bilinear"
    )

    # Repeat the dataset
    train_data = train_data.repeat()

    # Preprocess the data
    def preprocess(img):
        img = (tf.cast(img, "float32") - 127.5) / 127.5
        return img

    train = train_data.map(lambda x: preprocess(x))

    return train