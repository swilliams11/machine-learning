import os

import tensorflow as tf

"""
This class demonstrates the use to TFRecords, an efficient file format that uses protocol buffers. 
"""
print("Uncompressed TFRecords")
tf_record_dir = os.path.join("datasets", "tfrecord")
os.makedirs(tf_record_dir, exist_ok=True)

file_name = "my_data.tfrecord"
tf_record_path = os.path.join(tf_record_dir, file_name)

with tf.io.TFRecordWriter(tf_record_path) as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

filepaths = [tf_record_path]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)


# compresed TFRecords
print("Compressed TFRecords")
options = tf.io.TFRecordOptions(compression_type="GZIP")
tf_record_path = os.path.join(tf_record_dir, "my_compressed.tfrecord")
# open a TFRecordWriter and write a byte string to the file
with tf.io.TFRecordWriter(tf_record_path, options) as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

filepaths = [tf_record_path]
# open a TFRecordWriter and write a byte string to the file
dataset = tf.data.TFRecordDataset(filepaths, compression_type="GZIP")
for item in dataset:
    print(item)


# Protocol Buffers
print("Protocol Buffers")
# Generates "attempted relative import with no known parent package"
# from .proto.person_pb2 import Person
# person = Person(name="A1", id=123, email=["a@b.com"])
# print(person)

person_example = tf.train.Example(
    features=tf.train.Features(
        feature={
            "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"Alice"])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[123])),
            "emails": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"a@b.com", b"c@d.com"])),
        }
    )
)

print("Writing an example serialized object to disk.")
tf_record_path = os.path.join(tf_record_dir, "my_contacts.tfrecord")
with tf.io.TFRecordWriter(tf_record_path) as f:
    f.write(person_example.SerializeToString())

print("Reading the serialized object from disk.")
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

# This parses a single example at a time.
for serialized_example in tf.data.TFRecordDataset([tf_record_path]):
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

# Parse a batch of records at a time.
dataset = tf.data.TFRecordDataset([tf_record_path]).batch(10)
for serialized_examples in dataset:
    parsed_examples = tf.io.parse_example(serialized_examples, feature_description)








