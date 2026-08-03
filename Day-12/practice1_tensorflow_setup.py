"""
Day 12 - Practice 1: TensorFlow / Keras Installation Check

Before jumping into any deep learning code, it makes sense to first confirm
that TensorFlow is actually installed correctly and that Keras (which now
ships as part of TensorFlow) can be imported without issues. This small
script just does a sanity check.
"""

import sys


def check_tensorflow_installation():
    """Try importing TensorFlow and print basic environment info."""
    try:
        import tensorflow as tf
    except ImportError as error:
        print("TensorFlow is not installed in this environment.")
        print(f"Import error: {error}")
        print("Fix: run -> pip install tensorflow")
        sys.exit(1)

    print("TensorFlow imported successfully.")
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Keras version       : {tf.keras.__version__}")

    # Not every machine has a GPU, so this is just informational.
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        print(f"GPU detected ({len(gpu_devices)}): running on GPU is possible.")
    else:
        print("No GPU detected - TensorFlow will run on CPU, which is fine "
              "for the small models we're building in this internship.")

    return tf


def check_keras_import():
    """Confirm Keras submodules used later (layers, models) import fine."""
    try:
        from tensorflow.keras import layers, models  # 
    except ImportError as error:
        print(f"Could not import Keras layers/models: {error}")
        sys.exit(1)

    print("Keras 'layers' and 'models' submodules imported successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 12 - Practice 1: Verifying TensorFlow/Keras Setup")
    print("=" * 60)

    tf = check_tensorflow_installation()
    check_keras_import()

    print("\nSetup verified. Ready to move to Practice 2.")