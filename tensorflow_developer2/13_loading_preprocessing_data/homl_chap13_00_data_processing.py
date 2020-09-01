import tensorflow as tf
"""
This is chapter 13 of hands on machine learning revision 2. 
Data processing
"""

"""Creating and modifying a dataset"""
def dataset_from_tensor_slices():
    """Creates a dataset using from_tensor_slices."""
    X = tf.range(10)
    print(X)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    print(dataset)
    return dataset

def dataset_with_range():
    """Creates a dataset with the Dataset class Which is equivalent to the one above."""
    dataset = tf.data.Dataset.range(10)
    print(dataset)
    return dataset

def print_dataset(ds):
    # iterate over a dataset
    for item in ds:
        print(item)

def dataset_tansformation(ds):
    # repeat the dataset 3 times and then group into batch of 7
    return ds.repeat(3).batch(7)

def dataset_modification(ds):
    return ds.map(lambda x: x * 2)

def dataset_unbatch(ds):
    return ds.apply(ds.unbatch())

ds = dataset_from_tensor_slices()

"""
print("dataset from_tensor_slices")
print_dataset(dataset_from_tensor_slices())
print("dataset with Dataset.range()")
print_dataset(dataset_with_range())

print("\ndataset transformation")
ds2 = dataset_from_tensor_slices()
print_dataset(dataset_tansformation(ds2))

print("\ndataset modification")
print_dataset(dataset_modification(ds))
"""

print("\ndataset unbatched")
elements = [ [1, 2, 3], [1, 2], [1, 2, 3, 4] ]
dataset = tf.data.Dataset.from_generator(lambda: elements, tf.int64)
dataset = dataset.unbatch()
print(list(dataset.as_numpy_iterator()))

print("\nDataset apply function")
dataset = tf.data.Dataset.range(100)
def dataset_fn(ds):
  return ds.filter(lambda x: x < 5)
dataset = dataset.apply(dataset_fn)
print(list(dataset.as_numpy_iterator()))


print("\nDataset take function")
for item in dataset.take(3):
    print(item)

"""Shuffling a dataset"""

print("\nShuffling a dataset")
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
for item in dataset:
    print(item)

