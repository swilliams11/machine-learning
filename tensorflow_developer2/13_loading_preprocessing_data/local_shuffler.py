#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fnmatch
import os
import subprocess
import sys
import tarfile

import pandas as pd
import requests

__author__ = "Sean Williams"
__copyright__ = "Copyright 2020, Sean Williams"

"""
This module includes a Fetcher class to fetch a tgz file from an internet resource
and a Shuffle class to shuffle the data on the disk and split the files into training, testing and validation sets.
This module was created to test a portion of code in the Hands On Machine Learning v2 book - Chap 12.
Specifically, the section where it suggests that you can shuffle a file with linux commands before you 
shuffle it with TensorFlow as a best practice.   

By default is fetches the CA housing training dataset from the repo listed below.   
"""

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
FILE_DIRECTORY = "datasets/housing"
FILE_NAME = "housing.csv"


class Fetcher:
    """
    This class is responsible for fetching a tgz file from an internet resource and saving it
    to the local disk.
    """

    def __init__(self, download_url, extract_path, filename):
        self.download_url = download_url
        self.extract_path = extract_path
        self.filename = filename

    def fetch_housing_data(self):
        """Downloads the housing dataset from Github.
        Creates the directory in extract_path in the working directory.
        Extracts tarball from the tgz file into the extract_path directory.
        Fetches the data
        :return: nothing
        """
        os.makedirs(self.extract_path, exist_ok=True)
        try:
            r = requests.get(self.download_url, stream=True)
            if r.status_code is requests.codes.ok:
                with open(os.path.join(self.extract_path, self.filename), 'wb') as fd:
                    for chunk in r.iter_content(chunk_size=128):
                        fd.write(chunk)
            else:
                r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)
            sys.exit(1)

        tgz_path = os.path.join(self.extract_path, self.filename)
        if not tarfile.is_tarfile(tgz_path):
            print("The file that was download is not a tarfile; file type should be .tgz")
            sys.exit(1)
        try:
            housing_tgz = tarfile.open(tgz_path)
        except tarfile.TarError as err:
            print(err)
            sys.exit(1)
        housing_tgz.extractall(path=self.extract_path)
        housing_tgz.close()


class Shuffle:
    """
    This class is responsible for:
     1) shuffling the data on the disk by created a new shuffled file
     2) creating spliting the data into multiple files based on percentages (training, test, validation)
     3) spliting those files into multiple files as well.
    """

    def __init__(self, dataset_directory=None, dataset_filename=None, has_header_row=True):
        self.has_header_row = has_header_row
        self.dataset_directory = dataset_directory
        self.dataset_filename = dataset_filename
        self.dataset_file_location = os.path.join(self.dataset_directory, self.dataset_filename)
        self.shuffled_filename = self.rename_shuffled_file()
        self.shuffled_file_location = os.path.join(self.dataset_directory, self.shuffled_filename)
        self.percents = "80 10 10"

    def shuffle_on_disk(self):
        """
        Shuffles the data on disk using the linux `shuf` command.
        If has_header_row is True, then it will copy the header row to the new shuffled file,
        shuffle the dataset and then append the shuffled dataset to the shuffled file.
        If has_header_row is False, then it will shuffle the dataset and write it to a new file.
        It will automatically rename the dataset_filename (i.e. housing.txt) to housing_shuffled.txt.
        On MacOs you must execute `brew install coreutils` first.
        :return: None
        """
        number_to_skip = "+2"
        # self.dataset_file_location = os.path.join(self.dataset_directory, self.dataset_filename)
        # self.shuffled_filename = self.rename_shuffled_file()
        # self.shuffled_file_location = os.path.join(self.dataset_directory, self.shuffled_filename)
        print("Creating a new file and shuffling the source data {} file.".format(self.dataset_filename))
        cmd = "tail -n {} {} | shuf >> {}".format(number_to_skip, self.dataset_file_location,
                                                  self.shuffled_file_location)
        if self.has_header_row:
            copy_header_cmd = "head -n 1 {} >> {}".format(self.dataset_file_location, self.shuffled_file_location)
            os.system(copy_header_cmd)
        else:
            cmd.replace(number_to_skip, "")  # can ignore the header row, since it doesn't exist
            cmd.replace(">>", ">")  # change to write to non-existing file

        print(cmd)
        # os.system(cmd)
        subprocess.Popen(cmd, shell=True).wait()

        if not self._was_shuffle_successful():
            raise IOError("An error occurred during the shuffle. The file counts did not match.")

    def _was_shuffle_successful(self):
        """
        Checks if the `shuf` command was successful by comparing the line count for the original file and the
        shuffled file.
        :return: returns True if the file counts of the original file and the new shuffled file are the same,
        False otherwise.
        """
        line_count = "wc -l {}".format(self.dataset_file_location)
        # result = subprocess.check_output(["wc", "-l", dataset_file_location], shell=True)
        result = subprocess.check_output([line_count], shell=True)
        line_count2 = "wc -l {}".format(self.shuffled_file_location)
        result2 = subprocess.check_output([line_count2], shell=True)
        print("\nFile counts\n{}, {}".format(result, result2))
        return result.decode("utf-8").strip().split(" ", 1)[0] == result2.decode("utf-8").strip().split(" ", 1)[0]

    def rename_shuffled_file(self, dataset_filename=None):
        """
        Renames the shuffled file to append `_shuffled` to the file name.
        It splits the file name with one split and then appends `_shuffled`.
        :param dataset_filename: dataset file name
        :return: The renamed dataset file (i.e. housing.csv becomes housing_shuffled.csv)
        """
        split = self.dataset_filename.split(".", 1) if dataset_filename is None else dataset_filename.split(".", 1)
        renamed = split[0]  # take the first part of the file name
        ext = split[1]
        return renamed + "-shuffled." + ext

    def _add_parts_to_filename(self, dataset_filename=FILE_NAME, part_num=1):
        """
        Renames the filename to include a part extension.
        :param dataset_filename: dataset file name
        :return: The renamed dataset file (i.e. housing.csv becomes housing_shuffled.csv)
        """
        split = dataset_filename.split(".", 1)
        renamed = split[0]  # take the first part of the file name
        ext = split[1]
        return renamed + "-part" + str(part_num) + "." + ext

    def split_files(self, dataset_filename=None, percents=None):
        """
        Splits the dataset file into multiple files based on the percentages provided.
        If percents="80 20 20", then it will split the dataset file into 3 files based on those
        percentages.
        :param dataset_filename: will use the shuffled file name by default and the dataset_directory
        :param percents: A space separated list of the percents to split the file (.i.e. 60 20 20); it will
        use 80 10 10 by default
        :return: None
        """
        if percents is None:
            percents = self.percents
        if dataset_filename is None:
            dataset_filename = self.shuffled_filename

        split_cmd = "./split.sh " + self.dataset_directory + " " + dataset_filename + " " + percents
        subprocess.check_call(split_cmd, shell=True)
        if self.has_header_row:
            self._write_headers()

    def split_part_files(self, percents=None):
        """
        Splits the dataset part files into multiple files based on the percentages provided.
        If percents="80 20 20", then it will split the dataset part files into 3 files based on those
        percentages.  It searches for the files with `part` in their name.
        :param percents: A space separated list of the percents to split the file (.i.e. 60 20 20); it will
        use 80 10 10 by default
        :return: None
        """
        if percents is None:
            percents = self.percents

        files = os.listdir(self.dataset_directory)

        pattern = "*part*"
        for entry in files:
            if fnmatch.fnmatch(entry, pattern):
                print(entry)
                split_cmd = "./split.sh " + self.dataset_directory + " " + entry + " " + percents
                subprocess.check_call(split_cmd, shell=True)
        # TODO - this need to be modified to include part split or the main split.
        # if self.has_header_row:
        #    self._write_headers()

    def _write_headers(self):
        # shuffled_name = self.rename_shuffled_file(self.dataset_filename)
        headers = self._get_header_row()
        updated_name = self._add_parts_to_filename(self.shuffled_filename, 2)
        df = pd.read_csv(os.path.join(self.dataset_directory, updated_name), header=None)
        df.to_csv(os.path.join(self.dataset_directory, updated_name), header=headers.split(","), index=False)

        updated_name = self._add_parts_to_filename(self.shuffled_filename, 3)
        df = pd.read_csv(os.path.join(self.dataset_directory, updated_name), header=None)
        df.to_csv(os.path.join(self.dataset_directory, updated_name), header=headers.split(","), index=False)

    def _get_header_row(self):
        # newline = os.linesep  # Defines the newline based on your OS.
        source_fp = open(self.dataset_file_location, 'r')
        first_row = True
        for row in source_fp:
            if first_row:
                source_fp.close()
                return row.strip()


def fetch_and_shuffle_ca_housing_data():
    if not os.path.exists("datasets/housing/housing.tgz"):
        Fetcher(download_url=HOUSING_URL, extract_path=FILE_DIRECTORY, filename="housing.tgz").fetch_housing_data()
        shuffler = Shuffle(dataset_directory=FILE_DIRECTORY, dataset_filename=FILE_NAME)
        if not os.path.exists(shuffler.shuffled_file_location):
            shuffler.shuffle_on_disk()
        # split shuffled dataset into 3 files (train 80%, test 10%, validation 10%)
        shuffler.split_files(dataset_filename="housing-shuffled.csv")
        # split train, test, validation files into 3 files, so 9 new files are created.
        shuffler.split_part_files(percents="40 30 30")


if __name__ == "__main__":
    print("TODO - implement command line module")
    fetch_and_shuffle_ca_housing_data()