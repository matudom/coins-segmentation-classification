import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory

import pathlib


train_dir = input("Vyberte složku s trénovacími daty: \n")
train_dir = pathlib.Path(train_dir)


val_dir = input("Vyberte složku s validačními daty: \n")
val_dir = pathlib.Path(val_dir)

# Vytvoření datasetů
batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = image_dataset_from_directory(
  val_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(f"Třídy: {class_names}")

# Normalizace dat
normalization_layer = layers.Rescaling(1./255)

normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Vytvoření modelu
num_classes = len(class_names)

model = Sequential([
  layers.InputLayer(input_shape=(img_height, img_width, 3)),
  layers.Rescaling(1./255),
  layers.Conv2D(32, (3, 3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

model.summary()

# Trénování modelu
epochs = 15
history = model.fit(
  normalized_train_ds,
  validation_data=normalized_val_ds,
  epochs=epochs
)


np.save('class_names.npy', class_names)


model.save('coin_classifier_model.keras')