# TODO - implement tensorflow example code here.
import fnmatch
import os
import local_shuffler
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TensorFlowTraining2:

    def __init__(self, dataset_directory):
        self.training_pattern = "*train*"
        self.test_pattern = "*test*"
        self.valid_pattern = "*valid*"
        self.dataset_directory = dataset_directory
        self.train_file_paths = []
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.housing = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.train_filepaths = []
        self.valid_filepaths = []
        self.test_filepaths = []
        self.X_mean = None
        self.X_std = None
        self.model = None

    def _fetch_ca_datset(self):
        """
        Fetch the CA dataset and calculates the mean and standard deviation.
        Copied from HOML book.
        :return:
        """
        self.housing = fetch_california_housing()
        X_train_full, self.X_test, y_train_full, self.y_test = train_test_split(
            self.housing.data, self.housing.target.reshape(-1, 1), random_state=42)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_mean = scaler.mean_
        self.X_std = scaler.scale_

    def _save_to_multiple_csv_files(self, data, name_prefix, header=None, n_parts=10):
        """
        Save the files out to CSV files.
        Copied from HOML.
        :param name_prefix:
        :param header:
        :param n_parts:
        :return:
        """
        housing_dir = os.path.join("datasets", "housing")
        os.makedirs(housing_dir, exist_ok=True)
        path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

        filepaths = []
        m = len(data)
        for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
            part_csv = path_format.format(name_prefix, file_idx)
            filepaths.append(part_csv)
            with open(part_csv, "wt", encoding="utf-8") as f:
                if header is not None:
                    f.write(header)
                    f.write("\n")
                for row_idx in row_indices:
                    f.write(",".join([repr(col) for col in data[row_idx]]))
                    f.write("\n")
        return filepaths

    def _save_data_on_disk(self):
        train_data = np.c_[self.X_train, self.y_train]
        valid_data = np.c_[self.X_valid, self.y_valid]
        test_data = np.c_[self.X_test, self.y_test]
        header_cols = self.housing.feature_names + ["MedianHouseValue"]
        header = ",".join(header_cols)

        self.train_filepaths = self._save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
        self.valid_filepaths = self._save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
        self.test_filepaths = self._save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

    def fetch_and_save_data(self):
        self._fetch_ca_datset()
        if not os.path.exists(self.dataset_directory):
            self._save_data_on_disk()
        else:
            self._create_filepaths()

    def _create_filepaths(self):
        files = os.listdir(self.dataset_directory)
        for entry in files:
            if entry.find("train") >= 0:
                self.train_filepaths.append(os.path.join(self.dataset_directory, entry))
            elif entry.find("test") >= 0:
                self.test_filepaths.append(os.path.join(self.dataset_directory, entry))
            elif entry.find("valid") >= 0:
                self.valid_filepaths.append(os.path.join(self.dataset_directory, entry))

    @tf.function
    def preprocess(self, line):
        n_inputs = 8
        defaults = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
        fields = tf.io.decode_csv(line, record_defaults=defaults)
        x = tf.stack(fields[:-1])
        y = tf.stack(fields[-1:])
        return (x - self.X_mean) / self.X_std, y

    def _csv_reader_dataset(self, filepaths, repeat=1, n_readers=5, n_read_threads=None, shuffle_buffer_size=10000,
                            n_parse_threads=5, batch_size=32):

        dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
        dataset = dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
                                     cycle_length=n_readers,
                                     num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(self.preprocess, num_parallel_calls=n_parse_threads)
        return dataset.batch(batch_size).prefetch(1)

    def create_datasets(self):
        """
        Create the train, valid and test datasets.
        :return:
        """
        self.train_dataset = self._csv_reader_dataset(self.train_filepaths, repeat=None)
        self.valid_dataset = self._csv_reader_dataset(self.valid_filepaths)
        self.test_dataset = self._csv_reader_dataset(self.test_filepaths)

    def build_and_train_model(self):
        tf.keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(30, activation="relu", input_shape=self.X_train.shape[1:]),
            tf.keras.layers.Dense(1),
        ])

        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-3))
        batch_size = 32
        print(self.model.summary())
        self.model.fit(self.train_dataset, steps_per_epoch=len(self.X_train) // batch_size, epochs=10,
                  validation_data=self.valid_dataset)

    def save_model(self):
        self.model.save(os.path.join(self.dataset_directory, "ca_housing_data.h5"))

    def load_model(self):
        self.model = tf.keras.models.load_model(os.path.join(self.dataset_directory, "ca_housing_data.h5"))
        print("model loaded:")
        print(self.model.summary())

    def evaluate(self):
        batch_size = 32
        self.model.evaluate(self.test_dataset, steps=len(self.X_test) // batch_size)

    def predict(self):
        batch_size = 32
        new_set = self.test_dataset.map(lambda X, y: X)  # we could instead just pass test_set, Keras would ignore the labels
        X_new = self.X_test
        print(self.model.predict(new_set, steps=len(X_new) // batch_size))


tf_train = TensorFlowTraining2("datasets/housing")
tf_train.fetch_and_save_data()
tf_train.create_datasets()
my_model_dir = os.path.join(tf_train.dataset_directory, "ca_housing_data.h5")
if not os.path.exists(my_model_dir):
    "Building and training the model:"
    tf_train.build_and_train_model()
    tf_train.save_model()
else:
    tf_train.load_model()

print("Evaluating the model.")
tf_train.evaluate()
print("Predicting with the model.")
tf_train.predict()
