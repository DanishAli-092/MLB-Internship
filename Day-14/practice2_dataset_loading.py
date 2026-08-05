# Day 14 - Practice 2: Loading the Cats vs Dogs Dataset
#
# Just getting the dataset loaded and into a shape MobileNetV2 can
# actually take as input, before touching any model code.

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = (224, 224)   # MobileNetV2's expected input size
BATCH_SIZE = 32


def load_raw_dataset():
    # cats_vs_dogs only comes with one big 'train' split, no separate
    # val split - so slicing it 80/20 ourselves here.
    (train_ds, val_ds), info = tfds.load(
        "cats_vs_dogs",
        split=["train[:80%]", "train[80%:]"],
        with_info=True,
        as_supervised=True,   # returns (image, label) pairs instead of dicts
    )
    print(f"\nDataset info: {info.features}")
    print(f"Number of classes: {info.features['label'].num_classes}")
    print(f"Class names: {info.features['label'].names}")

    return train_ds, val_ds, info


def preprocess(image, label):
    # resize + cast to float32, that's it. Not scaling pixels here on
    # purpose - MobileNetV2's own preprocessing (scale to [-1,1]) gets
    # applied later inside the model, so this stays reusable elsewhere.
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def build_pipeline(dataset, shuffle=False):
    # map -> shuffle (train only) -> batch -> prefetch
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    train_ds_raw, val_ds_raw, info = load_raw_dataset()

    train_ds = build_pipeline(train_ds_raw, shuffle=True)
    val_ds = build_pipeline(val_ds_raw, shuffle=False)

    # Quick sanity check - grab one batch and print its shape
    for images, labels in train_ds.take(1):
        print(f"\nBatch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        print(f"Sample labels: {labels.numpy()[:10]}")