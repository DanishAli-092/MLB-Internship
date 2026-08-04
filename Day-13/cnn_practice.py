"""
Day 13 - CNN Practice (Practice 1, 2, 3)
-----------------------------------------
Goal for today was to get comfortable with the building blocks of a CNN
(Conv2D, MaxPooling, Flatten, Dense) before jumping into the full mini
project. This script covers:

    Practice 1 -> load Fashion MNIST, look at some samples, normalize it
    Practice 2 -> build + train a small CNN
    Practice 3 -> evaluate the model (accuracy, loss, sample predictions)

Everything is wrapped in small functions.
"""

import gzip
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# keep TF logs quiet, otherwise the console gets flooded with oneDNN /
# cpu-feature warnings that don't matter for a CPU run
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "practice")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed=SEED):
    """Fix seeds so re-runs are at least roughly reproducible."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        f.read(16)  # header:  count, rows, cols
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        f.read(8)  # header: magic, count
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def load_data():
    """
    Practice 1 - load Fashion MNIST.

    keras.datasets.fashion_mnist.load_data() pulls from Google Cloud
    Storage, which is blocked on the network I'm running this on, so I'm
    reading the raw idx files straight from ./data instead (same files,
    just parsed manually  it's the standard idx-ubyte format).
    """
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    for name, fname in files.items():
        full_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Missing dataset file: {full_path}. Download the Fashion-MNIST "
                f"gz files into the data/ folder first."
            )

    x_train = _read_idx_images(os.path.join(DATA_DIR, files["train_images"]))
    y_train = _read_idx_labels(os.path.join(DATA_DIR, files["train_labels"]))
    x_test = _read_idx_images(os.path.join(DATA_DIR, files["test_images"]))
    y_test = _read_idx_labels(os.path.join(DATA_DIR, files["test_labels"]))

    print(f"Train shape: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test shape:  {x_test.shape}, labels: {y_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def show_sample_images(x, y, n=10, save_path=None):
    """Practice 1 - plot n sample images with their class name as title."""
    plt.figure(figsize=(12, 5))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i], cmap="gray")
        plt.title(CLASS_NAMES[y[i]], fontsize=9)
        plt.axis("off")
    plt.suptitle("Fashion MNIST - sample training images")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved sample grid -> {save_path}")
    plt.close()


def preprocess(x_train, x_test):
    """
    Practice 1 - normalize pixel values to [0, 1] and add the channel
    dimension CNNs expect (28, 28) -> (28, 28, 1).
    """
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, x_test


def build_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Practice 2 - a small CNN, nothing fancy:
    Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Output
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),  # small dropout, dataset is easy enough to overfit fast
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, x_train, y_train, epochs=6, batch_size=128, val_split=0.1):
    """Practice 2 - train and keep 10% of train data for validation."""
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=2,
    )
    return history


def evaluate_model(model, x_test, y_test):
    """Practice 3 - report train/test accuracy + loss."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")
    return test_loss, test_acc


def show_predictions(model, x_test, y_test, n=10, save_path=None):
    """Practice 3 - run predictions on a handful of test images and plot them."""
    idx = np.random.choice(len(x_test), size=n, replace=False)
    preds = model.predict(x_test[idx], verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(12, 5))
    for i, sample_idx in enumerate(idx):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[sample_idx].squeeze(), cmap="gray")
        actual = CLASS_NAMES[y_test[sample_idx]]
        predicted = CLASS_NAMES[pred_labels[i]]
        color = "green" if predicted == actual else "red"
        plt.title(f"P: {predicted}\nA: {actual}", fontsize=8, color=color)
        plt.axis("off")
    plt.suptitle("Sample predictions (green = correct, red = wrong)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved prediction grid -> {save_path}")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed()

    print("=" * 60)
    print("PRACTICE 1: Loading + exploring the dataset")
    print("=" * 60)
    (x_train, y_train), (x_test, y_test) = load_data()
    show_sample_images(x_train, y_train, n=10,
                        save_path=os.path.join(OUTPUT_DIR, "sample_images.png"))
    x_train_norm, x_test_norm = preprocess(x_train, x_test)
    print(f"Pixel range after normalization: [{x_train_norm.min()}, {x_train_norm.max()}]")

    print("\n" + "=" * 60)
    print("PRACTICE 2: Building + training the CNN")
    print("=" * 60)
    model = build_cnn()
    model.summary()
    history = train_model(model, x_train_norm, y_train, epochs=6)

    print("\n" + "=" * 60)
    print("PRACTICE 3: Evaluating the model")
    print("=" * 60)
    train_acc = history.history["accuracy"][-1]
    print(f"Final training accuracy: {train_acc:.4f}")
    evaluate_model(model, x_test_norm, y_test)
    show_predictions(model, x_test_norm, y_test, n=10,
                      save_path=os.path.join(OUTPUT_DIR, "sample_predictions.png"))

    print("\nDone. Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # not swallowing the error silently, just make sure a stack trace
        # doesn't crash the whole terminal session without context
        print(f"\n[cnn_practice] Something went wrong: {exc}", file=sys.stderr)
        raise
