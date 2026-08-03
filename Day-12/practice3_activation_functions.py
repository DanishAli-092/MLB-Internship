"""
Day 12 - Practice 3: Experimenting with Activation Functions

Here we build the exact same network architecture three times, changing
only the activation function used in the hidden layer: ReLU, Sigmoid, and
Tanh. The architecture (number of layers/neurons) stays identical on
purpose, so the only thing that changes is how the hidden layer decides
what to "pass forward". This makes it easy to see that the activation
function does NOT change the model's structure (same number of params),
it only changes the math happening inside each neuron.
"""

import numpy as np
from tensorflow.keras import layers, models

ACTIVATIONS_TO_TEST = ["relu", "sigmoid", "tanh"]


def build_network_with_activation(activation_name: str, input_size: int = 20,
                                   hidden_units: int = 16, output_units: int = 4):
    """Build the same small ANN, swapping only the hidden layer activation."""
    model = models.Sequential(name=f"ANN_{activation_name}")
    model.add(layers.Input(shape=(input_size,)))
    model.add(layers.Dense(hidden_units, activation=activation_name,name=f"Hidden_{activation_name}"))
    model.add(layers.Dense(output_units, activation="softmax", name="Output_Layer"))
    return model


def inspect_activation_output(activation_name: str, sample_input: np.ndarray):
    """
    Manually pass a sample through just the hidden layer's activation so we
    can literally see how each function reshapes the same numbers.
    """
    layer = layers.Dense(6, activation=activation_name)
    output = layer(sample_input)
    return output.numpy()


if __name__ == "__main__":
    print("=" * 60)
    print("Day 12 - Practice 3: Comparing Activation Functions")
    print("=" * 60)

    # Same random input reused for every activation so the comparison is fair.
    np.random.seed(42)
    sample_input = np.random.uniform(-3, 3, size=(1, 6)).astype("float32")
    print(f"\nSample raw input values: {sample_input.flatten()}")

    for activation in ACTIVATIONS_TO_TEST:
        print(f"\n--- Activation: {activation.upper()} ---")

        try:
            model = build_network_with_activation(activation)
        except ValueError as error:
            print(f"Could not build model with '{activation}': {error}")
            continue

        total_params = model.count_params()
        print(f"Total trainable parameters: {total_params} "
              "(same for every activation - structure doesn't change)")

        output_values = inspect_activation_output(activation, sample_input)
        print(f"Output after activation:     {output_values.flatten()}")

    print("\nObservation:")
    print("  - ReLU    -> zeroes out negative values, keeps positives as-is")
    print("  - Sigmoid -> squashes everything into a (0, 1) range")
    print("  - Tanh    -> squashes everything into a (-1, 1) range")
    print("  - Parameter count stays identical across all three, confirming "
          "activation functions change behavior, not architecture.")