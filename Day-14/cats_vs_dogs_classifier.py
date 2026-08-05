# Day 14 - Mini Project: Cats vs Dogs Classifier using Transfer Learning
#
# Two-phase training: first freeze MobileNetV2 and just train the head
# (feature extraction), then unfreeze the top layers and fine-tune with
# a much lower learning rate. Target is 90%+ val accuracy, 93%+ if lucky.

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 5
FINE_TUNE_AT_LAYER = 100   # unfreeze from this layer index onward
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# 1. Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_datasets():
    (train_ds, val_ds), info = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        with_info=True,
        as_supervised=True,
    )
    class_names = info.features["label"].names
    return train_ds, val_ds, class_names


# Light augmentation so the model doesn't just memorize the training set.
# Keras only applies these during training, not at inference - handled
# automatically.
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def build_pipeline(dataset, shuffle=False):
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------------
# 2. Model building
# ---------------------------------------------------------------------------

def build_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # freeze for the first training phase

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train_feature_extraction(model, train_ds, val_ds):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
    )
    return history


def fine_tune(model, base_model, train_ds, val_ds):
    # unfreeze the top of MobileNetV2 and keep training, but with a much
    # smaller learning rate - otherwise we'd wreck the pretrained weights
    base_model.trainable = True

    # earlier layers stay frozen, only unfreeze from FINE_TUNE_AT_LAYER onward
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
    )
    return history_fine


# ---------------------------------------------------------------------------
# 4. Plotting and evaluation
# ---------------------------------------------------------------------------

def combine_histories(history, history_fine):
    acc = history.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss = history.history["loss"] + history_fine.history["loss"]
    val_loss = history.history["val_loss"] + history_fine.history["val_loss"]
    return acc, val_acc, loss, val_loss


def plot_curves(acc, val_acc, loss, val_loss):
    os.makedirs(f"{RESULTS_DIR}/graphs", exist_ok=True)
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.axvline(x=INITIAL_EPOCHS - 1, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.legend(loc="lower right")
    plt.title("Training vs Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.axvline(x=INITIAL_EPOCHS - 1, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.legend(loc="upper right")
    plt.title("Training vs Validation Loss")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graphs/accuracy_loss_curves.png")
    plt.close()
    print(f"Saved accuracy/loss curves to {RESULTS_DIR}/graphs/accuracy_loss_curves.png")


def show_sample_predictions(model, val_ds, class_names, num_samples=9):
    os.makedirs(f"{RESULTS_DIR}/predictions", exist_ok=True)

    images, labels = next(iter(val_ds.take(1)))
    predictions = model.predict(images)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    plt.figure(figsize=(10, 10))
    for i in range(min(num_samples, images.shape[0])):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[labels[i].numpy()]
        pred_label = class_names[predicted_labels[i]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/predictions/sample_predictions.png")
    plt.close()
    print(f"Saved sample predictions to {RESULTS_DIR}/predictions/sample_predictions.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_raw, val_raw, class_names = load_datasets()
    train_ds = build_pipeline(train_raw, shuffle=True)
    val_ds = build_pipeline(val_raw, shuffle=False)

    model, base_model = build_model()
    model.summary()

    print("\n--- Phase 1: Feature Extraction (base frozen) ---")
    history = train_feature_extraction(model, train_ds, val_ds)

    print("\n--- Phase 2: Fine-Tuning (top layers unfrozen) ---")
    history_fine = fine_tune(model, base_model, train_ds, val_ds)

    final_loss, final_acc = model.evaluate(val_ds)
    print(f"\nFinal Validation Accuracy: {final_acc * 100:.2f}%")
    print(f"Final Validation Loss: {final_loss:.4f}")

    acc, val_acc, loss, val_loss = combine_histories(history, history_fine)
    plot_curves(acc, val_acc, loss, val_loss)
    show_sample_predictions(model, val_ds, class_names)