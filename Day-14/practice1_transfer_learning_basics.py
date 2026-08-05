# Day 14 - Practice 1: Transfer Learning Basics
#
# Just messing around with MobileNetV2 here before actually training
# anything. Want to see what the architecture looks like, how many
# params it has, freeze it, and stick a small classifier head on top
# to check that everything connects properly.

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models


def load_pretrained_base(input_shape=(224, 224, 3)):
    # include_top=False drops MobileNetV2's original 1000-class ImageNet
    # head, since we're replacing it with our own head anyway.
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,   # drop the original 1000-class ImageNet head
        weights="imagenet"   # load pre-trained ImageNet weights
    )
    return base_model


def explore_architecture(base_model):
    # just checking how deep this thing actually is
    print("\n--- MobileNetV2 Architecture Summary ---")
    base_model.summary()

    total_layers = len(base_model.layers)
    total_params = base_model.count_params()

    print(f"\nTotal layers in MobileNetV2: {total_layers}")
    print(f"Total parameters: {total_params:,}")


def freeze_base_model(base_model):
    # trainable = False tells Keras to leave these weights alone during
    # backprop - only the layers we add on top should actually learn.
    base_model.trainable = False

    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in base_model.trainable_weights
    )
    print(f"\nTrainable parameters after freezing: {trainable_count}")
    return base_model


def build_model_with_custom_head(base_model, num_classes=2):
    # head is just: pool -> dense -> dropout -> output
    # GlobalAveragePooling2D squashes MobileNetV2's feature maps into a
    # single vector per image so it can go into normal Dense layers.
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # MobileNetV2 expects pixels scaled to [-1, 1], not [0, 255]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)  # keeps BatchNorm stats frozen too

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    base_model = load_pretrained_base()
    explore_architecture(base_model)
    base_model = freeze_base_model(base_model)

    full_model = build_model_with_custom_head(base_model, num_classes=2)

    print("\n--- Full Model (Base + Custom Head) Summary ---")
    full_model.summary()