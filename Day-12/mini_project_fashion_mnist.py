"""
Day 12 - Mini Project: First Artificial Neural Network
Dataset: Fashion MNIST (built into tensorflow.keras.datasets)

Workflow:
    1. Load the Fashion MNIST dataset
    2. Explore it (shapes, labels, a few sample images)
    3. Normalize pixel values to a 0-1 range
    4. Build a simple ANN (Input -> Hidden -> Output)
    5. Train the model, holding out part of the training data for validation
    6. Evaluate on the untouched test set
    7. Plot training/validation accuracy curves
    8. Run predictions on a handful of test images and visualize them

All generated plots are saved into the 'outputs' folder so they can be
included in the README / repo without needing to re-run training.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it works without a display, e.g. on a server
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data():
    """Load Fashion MNIST and print basic info about it."""
    print("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print(f"Training set: {x_train.shape[0]} images, each {x_train.shape[1]}x{x_train.shape[2]}")
    print(f"Test set:     {x_test.shape[0]} images, each {x_test.shape[1]}x{x_test.shape[2]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class names: {CLASS_NAMES}")

    return (x_train, y_train), (x_test, y_test)


def preview_sample_images(x_train, y_train, save_path):
    """Save a small grid of raw sample images just to eyeball the data."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i], cmap="gray")
        ax.set_title(CLASS_NAMES[y_train[i]], fontsize=9)
        ax.axis("off")
    fig.suptitle("Fashion MNIST - Sample Training Images")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved dataset preview -> {save_path}")


def normalize_pixels(x_train, x_test):
    """Scale pixel values from 0-255 down to 0-1, which helps training converge faster."""
    return x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0


def build_ann(input_shape=(28, 28), num_classes=10):
    """
    Build a simple ANN for image classification.

    Flatten turns each 28x28 image into a 784-length vector (since a plain
    Dense layer expects 1D input, not a 2D grid). Then it's the standard
    Input -> Hidden -> Output pattern from the practice scripts.
    """
    model = models.Sequential(name="Fashion_MNIST_ANN")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten(name="Flatten_Input"))
    model.add(layers.Dense(128, activation="relu", name="Hidden_Layer"))
    model.add(layers.Dropout(0.2, name="Dropout"))  # small regularization to reduce overfitting
    model.add(layers.Dense(num_classes, activation="softmax", name="Output_Layer"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_accuracy_curves(history, save_path):
    """Plot training vs validation accuracy across epochs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    ax.set_title("Fashion MNIST ANN - Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved accuracy curve -> {save_path}")


def plot_sample_predictions(model, x_test, y_test, save_path, num_samples=8):
    """Run predictions on a handful of test images and visualize predicted vs actual labels."""
    predictions = model.predict(x_test[:num_samples], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i], cmap="gray")
        actual = CLASS_NAMES[y_test[i]]
        predicted = CLASS_NAMES[predicted_labels[i]]
        is_correct = actual == predicted
        color = "green" if is_correct else "red"
        ax.set_title(f"Pred: {predicted}\nActual: {actual}", fontsize=9, color=color)
        ax.axis("off")
    fig.suptitle("Sample Predictions (green = correct, red = wrong)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved sample predictions -> {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        (x_train, y_train), (x_test, y_test) = load_data()
    except Exception as error:
        print(f"Failed to load dataset: {error}")
        return

    preview_sample_images(x_train, y_train, os.path.join(OUTPUT_DIR, "dataset_preview.png"))

    x_train, x_test = normalize_pixels(x_train, x_test)
    print("\nPixel values normalized to range [0, 1].")

    model = build_ann()
    print("\nModel Summary:")
    model.summary()

    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=2,
    )

    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal Training Accuracy:   {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    plot_accuracy_curves(history, os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
    plot_sample_predictions(model, x_test, y_test, os.path.join(OUTPUT_DIR, "sample_predictions.png"))

    # Save these numbers to a small text file so the README can reference
    # the exact figures without re-running training.
    results_path = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Final Training Accuracy:   {final_train_acc:.4f}\n")
        f.write(f"Final Validation Accuracy: {final_val_acc:.4f}\n")
        f.write(f"Test Accuracy:              {test_accuracy:.4f}\n")
        f.write(f"Test Loss:                  {test_loss:.4f}\n")
    print(f"\nSaved final results -> {results_path}")

    print("\nMini project complete.")


if __name__ == "__main__":
    main()