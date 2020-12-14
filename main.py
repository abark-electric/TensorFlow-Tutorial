"""
Object Detection From TF1 Saved Model
Algorithm Used: SSD MobileNet v2
"""

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
# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'    # Older version
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'    # Older version
# PATH_TO_MODEL_DIR = download_model(MODEL_NAME)
# print(PATH_TO_MODEL_DIR)


def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename, origin=base_url+filename, untar=False)
    label_dir = pathlib.Path(label_dir)

    return str(label_dir)


# LABEL_FILENAME = 'mscoco_label_map.pbtxt'
# PATH_TO_LABELS = download_labels(LABEL_FILENAME)

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = '/Users/khandokerabrarzawad/PycharmProjects/GitHub/TensorFlow-Tutorial/data/ssd_mobilenet_v1_coco_2017_11_17' + '/saved_model'
print('Loading model . . .', end='')
start_time = time.time()

# Load saved model and build the detection function
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = model.signatures['serving_default']

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Done! Took {elapsed_time} seconds.')

PATH_TO_LABELS = '/Users/khandokerabrarzawad/PycharmProjects/GitHub/TensorFlow-Tutorial/data/labels/mscoco_label_map.pbtxt'

# dict of index numbers to labels
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)


import numpy as np
from PIL import Image
import matplotlib
import warnings
#warnings.filterwarnings('ignore')
import PyQt5
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


def load_image_into_numpy_array(path):
    """
    Load an image from file into a numpy array.

    :param path:
    :return: uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


IMAGE_PATHS = ['/Users/khandokerabrarzawad/PycharmProjects/GitHub/TensorFlow-Tutorial/data/images/image1.jpg',
               '/Users/khandokerabrarzawad/PycharmProjects/GitHub/TensorFlow-Tutorial/data/images/image2.jpg']


for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Later
    # Flip image horizontally
    # image_np = np.fliplr(image_path).copy()

    # Later
    # Greyscale image
    # image_np = np.tile(np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(image_np)    # Convert input image to tensor
    input_tensor = input_tensor[tf.newaxis, ...]     # Add a batch of images, so adding an axis
    detections = detect_fn(input_tensor)    # Passing input image as a tensor through model

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.

    # Batch tensors essentially mean feeding multiple images at once
    # Our batch size here is 256
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    # print('Detections: \n')
    # print(detections)
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,    # mapping
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.9,
        agnostic_mode=False
    )
    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')



plt.show()
