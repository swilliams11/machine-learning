# TODO - implement tensoflow example code here.
import fnmatch
import os
import local_shuffler
import numpy as np
import pandas as pd
import tensorflow as tf


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


local_shuffler.fetch_and_shuffle_ca_housing_data()
