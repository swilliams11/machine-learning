import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""This module demonstrates keras api and shows that the saved model should reproduce the same results as the 
presaved model.  
https://www.tensorflow.org/guide/keras/train_and_evaluate#training_evaluation_from_tfdata_datasets
"""


def get_uncompiled_model():
    # Build a model using the functional API
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    my_model = keras.Model(inputs=inputs, outputs=outputs)
    return my_model


def get_compiled_model():
    my_model = get_uncompiled_model()
    my_model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Alternative way to compile a model.
    # my_model = model.compile(
    #     optimizer="rmsprop",
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )
    return my_model


def prepare_validation_set():
    # Prepare the validation dataset
    valid_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    valid_set = valid_set.batch(64)
    return valid_set

# print the model graph
#keras.utils.plot_model(model, "functional_api_model.png")
#keras.utils.plot_model(model, "functional_api_model_with_shapes.png", show_shapes=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# preprocess the data - convert the data type from uint8 to float32
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# take the last 10,000 records as the validation set
x_val = x_train[-10000:]
y_val = y_train[-10000:]
# take the first 50K records as the training dataset
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Create the datasets from Numpy arrays
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

val_dataset = prepare_validation_set()

model = get_compiled_model()

print(model.summary)
print("Training the model")
# This will loop through the entire dataset for each epoch
# history = model.fit(x_train, y_train, epochs=2)

# This will loop through a set number of records for each epoch; therefore, the dataset is not reset at the end of
# an epoch. In other words you may not loop through the entire dataset if you set this number too low.
# Also, you should expect longer training time to get better accuracy because it will not loop through the entire
# dataset for each epoch.
# history = model.fit(x_train, y_train, epochs=2, steps_per_epoch=100)

# Pass a validation set during training.
history = model.fit(x_train, y_train, epochs=2, validation_data=val_dataset)

print("Model History")
print(history.history)

print("Evaluate the test data")
results = model.evaluate(x_test, y_test, batch_size=64)
print(f"test loss, test accuracy:{results}")

# predict on the model
print("Predictions")
predictions = model.predict(x_test[:3])
print(f"predictions shape: {predictions.shape}")
print(predictions)

print("Saving the model")
model_name = "mnist_model_dataset"
model.save(model_name)

print("***** Testing the Saved Model *****")
print("Loading the model")
model_restored = keras.models.load_model(model_name)
print("Restored Model Summary")
print(model_restored.summary())

print("Evaluate the test data")
results2 = model_restored.evaluate(x_test, y_test, batch_size=64)
print(f"test loss, test accuracy:{results2}")

# predict on the model
print("Predictions")
predictions2 = model_restored.predict(x_test[:3])
print(f"predictions shape: {predictions2.shape}")
print(predictions2)




