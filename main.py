import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split


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


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0, 1_000])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"])
    plt.grid(True)
    plt.show()


def plot_xy(x, y, feature, labels):
    plt.scatter(feature, labels, label="Data")
    plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel("twitter_followers")
    plt.ylabel("popularity")
    plt.legend()
    plt.show()


def v3():
    uri = "./data/artists.csv"

    raw_dataset = pd.read_csv(
        uri,
        na_values="?",
        comment="\t",
        sep=",",
        skipinitialspace=True,
    )

    dataset = raw_dataset.copy()

    # dataset = dataset.dropna()
    # dataset = dataset[["twitter_followers", "popularity"]]
    dataset = dataset[["spotify_followers", "twitter_followers", "popularity"]]
    # followers_cut = 1_000_000
    # dataset = dataset[dataset["twitter_followers"] < followers_cut]
    dataset["popularity"] = dataset["popularity"] / 100
    dataset["twitter_followers"] = np.log2(dataset["twitter_followers"])

    print(dataset.tail())
    print(dataset.isna().sum())

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    def plot_values():
        sns.pairplot(
            train_dataset[
                [
                    # "spotify_followers",
                    "twitter_followers",
                    "popularity",
                ]
            ],
            diag_kind="kde",
        )
        plt.show()

    # plot_values()

    print(train_dataset.describe().transpose())

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop("popularity")
    test_labels = test_features.pop("popularity")

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    print(normalizer.mean.numpy())

    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print("First example:", first)
        print("Normalized:", normalizer(first).numpy())

    twitter_followers = np.array(train_features["twitter_followers"])

    twitter_followers_normalizer = layers.Normalization(
        input_shape=[
            1,
        ],
        axis=None,
    )
    twitter_followers_normalizer.adapt(twitter_followers)

    test_results = {}

    def one_variable():
        twitter_followers_model = tf.keras.Sequential(
            [twitter_followers_normalizer, layers.Dense(units=1)]
        )
        twitter_followers_model.summary()
        print(twitter_followers_model.predict(twitter_followers[:10]))

        twitter_followers_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss="mean_squared_error",
        )

        history = twitter_followers_model.fit(
            train_features["twitter_followers"],
            train_labels,
            epochs=2,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split=0.2,
        )

        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch
        print(hist.tail())

        test_loss = twitter_followers_model.evaluate(
            test_features["twitter_followers"], test_labels, verbose=2
        )
        print("\nTest loss:", test_loss)

        test_results["one_variable"] = test_loss

        # plot_loss(history)

        x = tf.linspace(0.0, np.log2(max(raw_dataset["twitter_followers"])), 1000)

        y = twitter_followers_model.predict(x)
        # plot_xy(x, y, train_features["twitter_followers"], train_labels)

        twitter_followers_model.save("./data/one_variable_model.keras")

    def multiple_variables():
        linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
        linear_model.predict(train_features[:10])
        print(linear_model.layers[1].kernel)

        linear_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss="mean_squared_error",
        )

        history = linear_model.fit(
            train_features,
            train_labels,
            epochs=2,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split=0.2,
        )

        hist = pd.DataFrame(history.history)
        hist["epoch"] = history.epoch
        print(hist.tail())

        test_loss = linear_model.evaluate(test_features, test_labels, verbose=2)
        print("\nTest loss:", test_loss)

        test_results["multiple_variables"] = test_loss

        # plot_loss(history)

    def deep_neural_network():
        def build_and_compile_model(norm):
            model = keras.Sequential(
                [
                    norm,
                    layers.Dense(64, activation="relu"),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(1),
                ]
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss="mean_squared_error",
            )

            return model

        dnn_twitter_followers_model = build_and_compile_model(
            twitter_followers_normalizer
        )

        dnn_twitter_followers_model.summary()

        history = dnn_twitter_followers_model.fit(
            train_features["twitter_followers"],
            train_labels,
            validation_split=0.2,
            verbose=0,
            epochs=4,
        )

        # plot_loss(history)

        # x = tf.linspace(0.0, 100_000_000, 1000)
        # y = dnn_twitter_followers_model.predict(x)
        # plot_xy(x, y, train_features["twitter_followers"], train_labels)

        test_loss = dnn_twitter_followers_model.evaluate(
            test_features["twitter_followers"], test_labels, verbose=2
        )
        print("\nTest loss:", test_loss)

        test_results["dnn"] = test_loss

        test_predictions = dnn_twitter_followers_model.predict(
            test_features["twitter_followers"]
        ).flatten()

        def plot_error():
            a = plt.axes(aspect="equal")
            plt.scatter(test_labels, test_predictions)
            plt.xlabel("True Values")
            plt.ylabel("Predictions")
            lims = [0, 1]
            plt.xlim(lims)
            plt.ylim(lims)
            _ = plt.plot(lims, lims)
            plt.show()

            error = test_predictions - test_labels
            plt.hist(error, bins=25)
            plt.xlabel("Prediction Error")
            _ = plt.ylabel("Count")
            plt.show()

        # plot_error()

    one_variable()
    # multiple_variables()
    # deep_neural_network()

    print(pd.DataFrame(test_results, index=["Mean square error"]).T)


def main():
    v3()


main()
