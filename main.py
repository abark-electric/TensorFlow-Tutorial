"""Object Detection From TF1 Saved Model"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')    # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_logical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def download_images():
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []
    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename, origin=base_url+filename, untar=False)
        image_path = pathlib.Path(image_path)
        image_paths.append((str(image_path)))
        print(image_paths)
    return image_paths


# IMAGE_PATHS = download_images()


def download_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = f'{model_name}.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url+model_file, untar=True)

    return str(model_dir)


# MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
# PATH_TO_MODEL_DIR = download_model(MODEL_NAME)


def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename, origin=base_url+filename, untar=False)
    label_dir = pathlib.Path(label_dir)

    return str(label_dir)


# LABEL_FILENAME = 'mscoco_label_map.pbtxt'
# PATH_TO_LABELS = download_labels(LABEL_FILENAME)
