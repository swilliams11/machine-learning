import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

"""
This class takes the fashion MNIST dataset, creates a training, validation and test set, 
builds a sequential model and predicts the results.  
"""

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=600):
    """
    Save the image.
    :param fig_id: image name
    :param tight_layout: True for tight layout and False otherwise
    :param fig_extension: image file extension
    :param resolution: image resolution
    :return:
    """
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Load the dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
# split the training set again into validation and training set
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# scale the RGB values and convert them to float
X_valid, X_train = X_valid / 255., X_train / 255.
X_test = X_test / 255.

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
save_fig("first_training_image")
#plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
               "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(X_train_full.shape)
print(type(X_train_full))
print(X_train.shape)
print(type(X_train))
model = keras.models.Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

print("Model Summary")
print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# history = model.fit(X_train_full, y_train_full, epochs=30, validation_data=(X_valid, y_valid))
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.ylim(bottom=0, top=1)
save_fig("keras_learning_curves_plot")
#plt.show()

model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])







