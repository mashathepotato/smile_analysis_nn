import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

import pathlib
from pathlib import Path

import os
from os import listdir


class ImageClassifier:
  def __init__(self, str_path, img_height=28, img_width=28, batch_size=32, epochs=10):
    self.str_path = str_path
    self.img_height = img_height
    self.img_width = img_width
    self.batch_size = batch_size
    self.epochs = epochs
    self.data_dir = pathlib.Path(self.str_path)
    self.train_ds, self.val_ds, self.class_names = self.load_data()
    self.num_classes = len(self.class_names)
    self.model = self.build_model()

  def load_data(self):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        self.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        self.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=self.batch_size
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

  def build_model(self):
    data_augmentation = keras.Sequential(
          [
              tf.keras.layers.RandomFlip("horizontal",
                                          input_shape=(self.img_height,
                                                      self.img_width,
                                                      3)),
              tf.keras.layers.RandomZoom(
                height_factor=(-0.2, -0.1),
                width_factor=None,
                fill_mode="reflect",
                interpolation="bilinear",
                seed=123),
              tf.keras.layers.RandomContrast(factor=(0.25, 0.75), seed=123)
          ]
      )
    
    model = Sequential([
      data_augmentation,
      tf.keras.layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      # layers.Conv2D(64, 3, padding='same', activation='relu'),
      # layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(32, activation="softmax", name="predictions"),
      layers.Dense(self.num_classes)
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

  def train_model(self):

    checkpoint_path = "models/sequential/cp.ckpt"
    checkpoint_dir = os.path.dirname(os.path.join("models/sequential", checkpoint_path))

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    history = self.model.fit(
        self.train_ds,
        validation_data=self.val_ds,
        epochs=self.epochs,
        callbacks=[cp_callback]
    )
    return history
  

  # TEST ACCURACY ON NEW IMAGES
  def test_accuracy(self):
    correct = 0
    total = 0
    for img in os.listdir("C:/Users/realc/OneDrive/Documents/IoM/Code/dataset/test"):
      img_dir = os.path.join("C:/Users/realc/OneDrive/Documents/IoM/Code/dataset/test", img)
      img_dir = pathlib.Path(img_dir)

      img = tf.keras.utils.load_img(
        img_dir, target_size=(self.img_height, self.img_width)
      )

      img_array = tf.keras.utils.img_to_array(img)
      img_array = tf.expand_dims(img_array, 0)  # Create a batch

      predictions = self.model.predict(img_array)
      score = tf.nn.softmax(predictions[0])

      if str(self.class_names[np.argmax(score)]) in str(img_dir):
        correct += 1
      total += 1

    final_accuracy = (correct / total) * 100
    print("Total testing: ", total)
    return final_accuracy
  

  def analyze_image(self, img):
    img_dir = Path(img)

    img = tf.keras.utils.load_img(
      img_dir, target_size=(self.img_height, self.img_width)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = self.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    result = str(self.class_names[np.argmax(score)])

    if "gummy" in result:
      print("Excessive gingival display detected")
    else:
      print("No excessive gingival dispay detected")

    return result
  
  def analyze_image(self):
    img_dir = Path("cropped_captured_image.jpg")

    img = tf.keras.utils.load_img(
      img_dir, target_size=(self.img_height, self.img_width)
    )  

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    self.model.load_weights("models/sequential/cp.ckpt").expect_partial()

    predictions = self.model.predict(img_array)
    # predictions = self.model.load_weights("models/sequential/cp.ckpt")
    score = tf.nn.softmax(predictions[0])

    result = str(self.class_names[np.argmax(score)])

    if "gummy" in result:
      return True
    
    return False


  def visualize_training(self, history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(self.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("% Accuracy")  
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("% Accuracy")  
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == "__main__":
  str_path = "C:/Users/realc/OneDrive/Documents/IoM/Code/dataset/gum"
  img_classifier = ImageClassifier(str_path)


  # history = img_classifier.train_model()
  # img_classifier.model.summary()

  # print(img_classifier.test_accuracy())

  # img_classifier.visualize_training(history)

  # CLASSIFY CUSTOM IMAGE
  print(img_classifier.analyze_image())

  if img_classifier.analyze_image():
    print("Gummy smile detected")
  else:
    print("Normal smile")