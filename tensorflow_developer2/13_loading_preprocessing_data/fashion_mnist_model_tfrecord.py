import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Flatten, Dense, LayerNormalization
from contextlib import ExitStack
import IPython.display as display
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


"""
This class takes 
1) the fashion MNIST dataset, creates a training, validation and test set, 
2) saves the data to TFRecord
3) builds a sequential model and predicts the results.  
4) loads the saved model if one exists
5) Executes evaluate on the saved model
6) saved the predicted results as an image. 
"""

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
BATCH_SIZE = 32


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=900):
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


def create_proto_buffer(image, label):
    Example = tf.train.Example
    Features = tf.train.Features
    Feature = tf.train.Feature

    image_string = tf.io.serialize_tensor(image)
    #image_string = keras.backend.eval(image_string)
    mnist_image = Example(
        features=Features(
            feature={
                "image": Feature(bytes_list=tf.train.BytesList(value=[image_string.numpy()])),
                "label": Feature(int64_list=tf.train.Int64List(value=[label]))
            }
        )
    )
    return mnist_image


def write_tfrecords(name, dataset, n_shards=10):
    """
    This is from HOML example.
    :param name: file name and location
    :param dataset: TensorFlow dataset to split into multiple shards
    :param n_shards: The number of shard to split the dataset into
    :return: A list of file paths
    """
    # Create the file name path with list comprehensions
    paths = ["{}.tfrecord-{:05d}-of-{:05d}".format(name, index, n_shards)
             for index in range(n_shards)]
    # ExitStack closes all open files at the end of the with statement
    with ExitStack() as stack:
        # Create a list of paths
        writers = [stack.enter_context(tf.io.TFRecordWriter(path))
                   for path in paths]
        # loop through each item in the dataset and write to a file
        for index, (image, label) in dataset.enumerate():
            shard = index % n_shards
            example = create_proto_buffer(image, label)
            writers[shard].write(example.SerializeToString())
    return paths


# Create the feature description
feature_descr = {
    "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1)
}

def normalize_test_data(dataset):
    X_train = None
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_mean = scaler.mean_
    X_std = scaler.scale_

def parse_example_proto(example_proto):
    """
    Read the single example back.
    :param example_proto:
    :return: Returns a dictionary of tensors
    """
    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_descr)
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
    # image = tf.io.decode_jpeg(example["image"])
    image = tf.reshape(image, shape=[28, 28])
    return image, example["label"]


def parse_example_proto_test_predict(example_proto):
    """
    Read the single example back for testing and prediction data.
    :param example_proto:
    :return: Returns a dictionary of tensors
    """
    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_descr)
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
    # image = tf.io.decode_jpeg(example["image"])
    image = tf.reshape(image, shape=[28, 28])
    image = tf.cast(image, tf.float32) / 255.
    # TODO - normalize the data here as well (so calc the mean and standard deviation)
    return image, example["label"]


def dataset_reader(filepaths, repeat=1, n_readers=5, n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32, training=True):
    """
    Create a TensorFlow dataset object from the filepaths.
    :param filepaths: List of file paths/patterns
    :param repeat: repeat this dataset so each original value is seen "repeat" times
    :param n_readers: cycle_length controls the number of input elements that are processed concurrently
    :param n_read_threads: num_parallel_calls - makes interleave use multiple threads to fetch elements.
    :param shuffle_buffer_size: The dataset will fill a buffer of buffer_size elements.
    :param n_parse_threads: the number of threads to execute in parallel
    :param batch_size: sets the size of the batch.
    :return:
    """
    #dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = filepaths.repeat(repeat)
    # maps a function across this dataset and interleaves the results
    dataset = dataset.interleave(lambda filepath: tf.data.TFRecordDataset(filepath),
                                 cycle_length=n_readers,
                                 num_parallel_calls=n_read_threads)

    # map the preprocessing function onto the dataset.
    if training:
        # Randomly shuffles the elements of this dataset, draws 1 element from buffer and replaces it with new element
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(parse_example_proto, num_parallel_calls=n_parse_threads)
    else:
        dataset = dataset.map(parse_example_proto_test_predict, num_parallel_calls=n_parse_threads)
    # combines the dataset elements into batches
    # prefetch fetches 1 batch of 32 records.
    return dataset.batch(batch_size).prefetch(1)


def create_file_paths(data_dir, file_pattern, shuffle):
    mnist_file = os.path.join(data_dir, file_pattern)
    return tf.data.Dataset.list_files(mnist_file, shuffle=shuffle)


# Load the dataset, split it into training, validation and test sets and shuffle the files.
ds_train, ds_validation, ds_test = tfds.load(name="fashion_mnist", split=["train[:50000]", "train[50000:]", "test"],
                                             shuffle_files=True, as_supervised=True)

# Shuffle the training set
ds_train_shuffled = ds_train.shuffle(50000)
mnist_dir = "datasets/fashion_mnist"
mnist_file_name = "fashion_mnist.tfrecord"
os.makedirs(mnist_dir, exist_ok=True)
mnist_file_location = os.path.join(mnist_dir, mnist_file_name)

# loop through the dataset and display the image.
# for image_features in parsed_image_dataset:
#   image_raw = tf.io.parse_tensor(image_features['image'], out_type=tf.uint8)
#   image_raw2 = tf.reshape(image_raw, shape=[28, 28])
#   #display.display(display.Image(data=image_raw))
#   _ = plt.imshow(image_raw2, cmap="binary")

# Write the data to the datasets directory
if not os.path.exists(os.path.join(mnist_dir, "fashion_mnist_test.tfrecord.tfrecord-00000-of-00010")):
    print(f"Writing shards of data to {mnist_dir}")
    file_name = os.path.join(mnist_dir, "fashion_mnist_train.tfrecord")
    write_tfrecords(file_name, ds_train)
    file_name = os.path.join(mnist_dir, "fashion_mnist_valid.tfrecord")
    write_tfrecords(file_name, ds_validation)
    file_name = os.path.join(mnist_dir, "fashion_mnist_test.tfrecord")
    write_tfrecords(file_name, ds_test)

# Create the file paths
train_filepaths = create_file_paths(mnist_dir, "*train*", shuffle=True)
test_filepaths = create_file_paths(mnist_dir, "*test*", shuffle=False)
valid_filepaths = create_file_paths(mnist_dir, "*valid*", shuffle=False)

# Read the sharded data back
train_dataset = dataset_reader(train_filepaths, repeat=None, shuffle_buffer_size=50000)
valid_dataset = dataset_reader(valid_filepaths, training=False)
test_dataset = dataset_reader(test_filepaths, training=False)
result = test_dataset.element_spec
result2 = result[0]
# map the
# image_dataset = tf.data.TFRecordDataset(mnist_file_location)
# parsed_image_dataset = image_dataset.map(parse_example_proto)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
               "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Build and train the model
model_dir = "models"
model_name = "fashion_minst_model"
history_name = "fashion_minst_history.npy"
model_location = os.path.join(model_dir, model_name)
history_location = os.path.join(model_dir, history_name)
model = None
history = None

#for num, _ in enumerate(train_dataset):
#    pass

if not os.path.exists(model_location):
    print("Building the model.")
    os.makedirs(model_dir, exist_ok=True)
    model = keras.models.Sequential()
    model.add(LayerNormalization(input_shape=[28, 28]))
    model.add(Flatten(input_shape=[28, 28]))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.build()
    history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset, steps_per_epoch=50000 // BATCH_SIZE)
    #history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset)
    np.save(history_location, history.history)
    model.save(model_location)
else:
    print("Loading the saved model.")
    history = np.load(history_location, allow_pickle='TRUE').item() if os.path.exists(history_location) else None
    model = tf.keras.models.load_model(model_location)
    # This is the line the fixed my low accuracy on the saved test set. You have to call model compile.
    # The documentation states that the model is already compiled.
    # Once I added this line the accuracy went from 10% to 87%
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# print("Model weights")
# print(model.weights)
print("Model Summary")
print(model.summary())

history_eval = model.evaluate(test_dataset, batch_size=32)
print(history_eval)

# one Dataset item has 32 records
for X, y in test_dataset.take(1):
    for i in range(5): # only loop through the first 5 records
        plt.subplot(1, 5, i + 1)
        plt.imshow(X[i].numpy(), cmap="binary")
        plt.axis("off")
        plt.title(str(y[i].numpy()) + ' ' + str(class_names[y[i].numpy()]))
    save_fig("fashion_mnist_test.png")

X_new = test_dataset.take(1)
y_proba = model.predict(X_new)
y_proba.round(2)
y_pred = model.predict_classes(X_new)
#new_array = np.array(zip(y_pred, class_names[y_pred]))
print("Predictions")
print(y_pred)
print("Predicted Class Names")
for i in y_pred:
    print(class_names[i])

# Save the predicted class names and the images to a file.
for X, y in X_new:
    for i in range(32):
        plt.subplot(7, 5, i + 1)
        plt.imshow(X[i].numpy(), cmap="binary")
        plt.axis("off")
        plt.title(str(y[i].numpy()) + ' ' + str(class_names[y[i].numpy()]))
    save_fig("fashion_mnist_predictions", tight_layout=False)