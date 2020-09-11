import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""This module demonstrates keras api and shows that the saved model should reproduce the same results as the 
presaved model.  
https://www.tensorflow.org/guide/keras/train_and_evaluate#training_evaluation_from_tfdata_datasets
"""

# Build a model using the functional API
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# print the model graph
#keras.utils.plot_model(model, "functional_api_model.png")
#keras.utils.plot_model(model, "functional_api_model_with_shapes.png", show_shapes=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# preprocess the data
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
print(model.summary)
print("Training the model")
#history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
history = model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
print("Model History")
print(history.history)

print("Evaluate the test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print(f"test loss, test accuracy:{results}")

# predict on the model
print("Predictions")
predictions = model.predict(x_test[:3])
print(f"predictions shape: {predictions.shape}")
print(predictions)

print("Saving the model")
model.save("mnist_model")

print("***** Testing the Saved Model *****")
print("Loading the model")
model_restored = keras.models.load_model("mnist_model")
print("Restored Model Summary")
print(model_restored.summary())

print("Evaluate the test data")
results2 = model_restored.evaluate(x_test, y_test, batch_size=128)
print(f"test loss, test accuracy:{results2}")

# predict on the model
print("Predictions")
predictions2 = model_restored.predict(x_test[:3])
print(f"predictions shape: {predictions2.shape}")
print(predictions2)




