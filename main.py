import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(items):
    data = items[1:, :-1]
    data = data.astype("int32")

    labels = items[1:, -1]
    labels = labels.astype("int32")

    return train_test_split(data, labels, test_size=0.2)


def v1():
    arr = np.loadtxt("./data/artists.v1.csv", delimiter=",", dtype=str)

    train_items, test_items, train_labels, test_labels = load_data(arr)

    print("train_items.shape", train_items.shape)
    print("train_labels", train_labels.shape)

    print("test_items.shape", test_items.shape)
    print("test_labels", test_labels.shape)

    features = train_items.shape[1]

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(features,)),
            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_initializer="he_normal",
            ),
            tf.keras.layers.Dense(8, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_items, train_labels, epochs=150, batch_size=32, verbose=0)

    test_loss, test_acc = model.evaluate(test_items, test_labels, verbose=2)

    print("\nTest accuracy:", test_acc)

    artist = test_items[0:1]
    predictions = model.predict([artist])
    print(predictions, test_labels[0:1])


def v2():
    arr = np.loadtxt("./data/artists.csv", delimiter=",", dtype=str)

    train_items, test_items, train_labels, test_labels = load_data(arr)

    print("train_items.shape", train_items.shape)
    print("train_labels", train_labels.shape)

    print("test_items.shape", test_items.shape)
    print("test_labels", test_labels.shape)

    features = train_items.shape[1]

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(features,)),
            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_initializer="he_normal",
            ),
            # tf.keras.layers.Dense(8, activation="relu", kernel_initializer="he_normal"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_items, train_labels, epochs=150, batch_size=32, verbose=0)

    test_loss, test_acc = model.evaluate(test_items, test_labels, verbose=2)

    print("\nTest accuracy:", test_acc)

    artist = test_items[0:1]
    predictions = model.predict([artist])
    print("predictions:", predictions, "test_labels:", test_labels[0:1])


def main():
    v1()


main()
