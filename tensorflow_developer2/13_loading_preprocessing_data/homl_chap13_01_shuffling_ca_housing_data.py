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

class TensorFlowTraining():
    train_file_pattern = "*part1*part*"
    train_file_pattern_for_mean_std = "*part1.csv"

    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.train_file_paths = []
        self.dataset = None
        nonlocal train_file_pattern
        self.training_files_location = os.path.join(self.dataset_directory, train_file_pattern)
        self.dataset_mean = []
        self.dataset_std = []
        self.housing = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.train_filepaths = None
        self.valid_filepaths = None
        self.test_filepaths = None
        self.X_mean = None
        self.X_std = None


    def _create_train_file_paths(self, train_pattern_override=None):
        """
        This method creates an array
        :param train_pattern_override:
        :return:
        """
        nonlocal train_file_pattern
        training_pattern = train_file_pattern if train_pattern_override is None else train_pattern_override
        files = os.listdir(self.dataset_directory)
        for entry in files:
            if fnmatch.fnmatch(entry, training_pattern):
                self.train_file_paths.append(entry)

    def _create_train_file_dataset(self, train_pattern_override=None):
        nonlocal train_file_pattern
        training_pattern = train_file_pattern if train_pattern_override is None else train_pattern_override
        training_files_location = self.training_files_location if train_pattern_override is None \
            else os.path.join(self.dataset_directory, train_pattern_override)
        # list_files shuffles the file paths pass seed=42 to shuffle same way each time.
        self.dataset = tf.data.Dataset.list_files(self.training_files_location)

    def preprocess(self, line):
        n_inputs = 8
        defaults = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
        fields = tf.io.decode_csv(line, records_defaults=defaults)
        x = tf.stack(fields[:-1])
        y = tf.stack(fields[-1:])
        return (x - self.dataset_mean) / self.dataset_std, y

    def calc_mean_std_dv(self):
        """
        Calculates the mean and standard deviation of the training file.
        :return:
        """
        nonlocal train_file_pattern_for_mean_std
        training_file_location = os.path.join(self.dataset_directory, train_file_pattern_for_mean_std)
        dataset = pd.read_csv(training_file_location)
        # only calculates std and mean for numeric values
        std: pd.Series = dataset.std()
        mean: pd.Series = dataset.mean()
        # we only need the first 8 items, because the 9th one is the predicted value
        self.dataset_mean = np.array(mean.array[:-1], dtype="float32")
        self.dataset_std = np.array(std.array[:-1], dtype="float32")

    def csv_reader_dataset(self, repeat=1, n_readers=5, n_read_threads=None, shuffle_buffer_size=10000,
                           n_parse_threads=5, batch_size=32):

        dataset = self.dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
                                          cycle_length=n_readers,
                                          num_parallel_calls=n_read_threads)
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
        dataset = dataset.map(self.preprocess(), num_parallel_calls=n_parse_threads)
        return dataset.batch(batch_size).prefetch(1)

    def fetch_ca_datset(self):
        """
        Fetch the CA dataset and calculates the mean and standard deviation.
        Copied from HOML book.
        :return:
        """
        self.housing = fetch_california_housing()
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.housing.data, self.housing.target.reshape(-1, 1), random_state=42)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X_train_full, y_train_full, random_state=42)

        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_mean = scaler.mean_
        self.X_std = scaler.scale_

    def write_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
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


    def save_data_on_disk(self):
        train_data = np.c_[self.X_train, self.y_train]
        valid_data = np.c_[self.X_valid, self.y_valid]
        test_data = np.c_[self.X_test, self.y_test]
        header_cols = self.housing.feature_names + ["MedianHouseValue"]
        header = ",".join(header_cols)

        self.train_filepaths = self.save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
        self.valid_filepaths = self.save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
        self.test_filepaths = self.save_to_multiple_csv_files(test_data, "test", header, n_parts=10)


