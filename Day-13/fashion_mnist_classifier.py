"""
Day 13 Mini Project - Fashion MNIST Image Classifier using CNN
----------------------------------------------------------------
Builds on cnn_practice.py but goes further: trains a slightly deeper CNN,
tracks train/val accuracy and loss curves, plots a confusion matrix, and
pulls out 10 correctly + 10 incorrectly classified test images so I can
actually see where the model is getting confused (mostly shirt vs
pullover vs coat, as expected - they look similar even to me at 28x28).

Run:
    python fashion_mnist_classifier.py
"""

import gzip
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "project")
MODEL_PATH = os.path.join(OUTPUT_DIR, "fashion_cnn.keras")

EPOCHS = 12
BATCH_SIZE = 128

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# Data loading

def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        f.read(16)
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        f.read(8)
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def load_fashion_mnist():
    """
    Loads Fashion MNIST from local gz files in data/.
    (Skipping keras.datasets.fashion_mnist.load_data() here since it hits
    Google Cloud Storage, which isn't reachable from this machine same
    dataset either way, just parsing the idx-ubyte format by hand.)
    """
    required = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
    ]
    for fname in required:
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            raise FileNotFoundError(
                f"Missing {fname} in {DATA_DIR}. Grab the Fashion-MNIST gz "
                f"files and drop them in data/ before running this script."
            )

    x_train = _read_idx_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
    x_test = _read_idx_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
    y_test = _read_idx_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
    return (x_train, y_train), (x_test, y_test)


def preprocess(x_train, x_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, x_test


# Visualization helpers


def plot_sample_grid(x, y, title, save_path, n=10, preds=None):
    plt.figure(figsize=(12, 5))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i].squeeze(), cmap="gray")
        if preds is None:
            plt.title(CLASS_NAMES[y[i]], fontsize=9)
        else:
            actual, predicted = CLASS_NAMES[y[i]], CLASS_NAMES[preds[i]]
            color = "green" if actual == predicted else "red"
            plt.title(f"P: {predicted}\nA: {actual}", fontsize=8, color=color)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy per epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss per epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Fashion MNIST CNN")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")
    return cm


def plot_correct_incorrect(x_test, y_test, y_pred, save_path_correct, save_path_incorrect, n=10):
    correct_idx = np.where(y_test == y_pred)[0]
    incorrect_idx = np.where(y_test != y_pred)[0]

    rng = np.random.default_rng(SEED)
    correct_sample = rng.choice(correct_idx, size=min(n, len(correct_idx)), replace=False)
    incorrect_sample = rng.choice(incorrect_idx, size=min(n, len(incorrect_idx)), replace=False)

    def _plot(indices, title, path):
        plt.figure(figsize=(12, 5))
        for i, idx in enumerate(indices):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x_test[idx].squeeze(), cmap="gray")
            actual, predicted = CLASS_NAMES[y_test[idx]], CLASS_NAMES[y_pred[idx]]
            plt.title(f"P: {predicted}\nA: {actual}", fontsize=8)
            plt.axis("off")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {path}")

    _plot(correct_sample, "10 correctly classified images", save_path_correct)
    _plot(incorrect_sample, "10 incorrectly classified images", save_path_incorrect)
    return len(correct_idx), len(incorrect_idx)



# Model

def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Slightly deeper than the practice model - added a 3rd conv block and
    batch norm since I noticed the practice model's val_loss was still
    inching down at epoch 6, figured a bit more capacity + more epochs
    would help without badly overfitting (dataset is small/simple enough
    that dropout + batchnorm keeps it in check).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Main pipeline

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading Fashion MNIST...")
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    plot_sample_grid(x_train, y_train, "Fashion MNIST - training samples",
                      os.path.join(OUTPUT_DIR, "01_sample_images.png"))

    x_train_n, x_test_n = preprocess(x_train, x_test)

    print("\nBuilding model...")
    model = build_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    print(f"\nTraining for up to {EPOCHS} epochs (early stopping on val_loss)...")
    history = model.fit(
        x_train_n, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    plot_training_curves(history, os.path.join(OUTPUT_DIR, "02_training_curves.png"))

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test_n, y_test, verbose=0)
    final_train_acc = history.history["accuracy"][-1]
    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Test accuracy:        {test_acc:.4f}")
    print(f"Test loss:            {test_loss:.4f}")

    print("\nRunning predictions on full test set...")
    probs = model.predict(x_test_n, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    plot_sample_grid(x_test_n, y_test, "Predictions on sample test images",
                      os.path.join(OUTPUT_DIR, "03_sample_predictions.png"),
                      n=10, preds=y_pred)

    cm = plot_confusion_matrix(y_test, y_pred, os.path.join(OUTPUT_DIR, "04_confusion_matrix.png"))

    n_correct, n_incorrect = plot_correct_incorrect(
        x_test_n, y_test, y_pred,
        os.path.join(OUTPUT_DIR, "05_correctly_classified.png"),
        os.path.join(OUTPUT_DIR, "06_incorrectly_classified.png"),
    )
    print(f"\nCorrect on test set:   {n_correct}/{len(y_test)}")
    print(f"Incorrect on test set: {n_incorrect}/{len(y_test)}")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print("\nClassification report:\n", report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Final train accuracy: {final_train_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n\n")
        f.write(report)

    model.save(MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")
    print("All outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
