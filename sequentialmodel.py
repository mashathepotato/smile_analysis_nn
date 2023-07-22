import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
import os


from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

import pathlib
from pathlib import Path

import os
from os import listdir

import matplotlib.pyplot as plt

# ROSES EXAMPLE
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# print(data_dir)
# data_dir = pathlib.Path(data_dir)
# print(data_dir)


str_path = "C:/Users/realc/OneDrive/Documents/IoM/Code/dataset"
# str_path = "C:/things/teeth/dataset/flat"
data_dir = pathlib.Path(str_path) # Path type
# data_dir = os.listdir(str_path)

class_names = ["flat", "ideal", "inverted"]

# APPEND EVERY IMAGE TO TEETH LIST
# teeth = []
# # get the path/directory
# for im_class in class_names:
#   for image in os.listdir(str_path + "/" + im_class):
#       if (image.endswith(".jpg")):
#           teeth.append(image)
# print(teeth)


# SEPARATE IMAGES INTO LISTS BY CLASS
# ideal = list(data_dir.glob('ideal/*'))
# # print(ideal)
# im = Image.open(ideal[0])
# # im.show()

# teeth = list(data_dir.glob('dataset/flat'))
# print(teeth)
# print(len(list(teeth)))
# print(teeth[0])

# LOAD DATA
batch_size = 10
img_height = 28
img_width = 28

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
  )

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
  )

class_names = train_ds.class_names
# print(class_names)


# SHOW FIRST n IMAGES
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")
# plt.show()


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)


# BUILD MODEL
model = Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# COMPILE MODEL
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.summary()
