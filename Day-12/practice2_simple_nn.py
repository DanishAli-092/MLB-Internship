"""
Day 12 - Practice 2: Build a Simple Neural Network

Goal: build the smallest possible ANN that still has all three core pieces
of a neural network:
    1. Input Layer   - just receives the raw features, no computation here
    2. Hidden Layer  - learns patterns from the input using weights + activation
    3. Output Layer  - produces the final prediction

We're not training on real data yet (that's the mini project). The point of
this script is purely to understand how the layers connect and what the
model summary is actually telling us.
"""

from tensorflow.keras import layers, models


def build_simple_network(input_size: int, hidden_units: int, output_units: int):
    """
    Build a basic feed-forward ANN: Input -> Dense (hidden) -> Dense (output).

    Parameters
    ----------
    input_size : int
        Number of features coming into the network (size of input layer).
    hidden_units : int
        Number of neurons in the single hidden layer.
    output_units : int
        Number of neurons in the output layer (e.g. number of classes).

    Returns
    -------
    keras.Model
        Compiled-ready (but not yet compiled) sequential model.
    """
    model = models.Sequential(name="Simple_ANN")

    # Input layer: doesn't "do" anything mathematically, it just defines
    # the shape of data the network should expect.
    model.add(layers.Input(shape=(input_size,), name="Input_Layer"))

    # Hidden layer: this is where the actual learning happens. ReLU is used
    # here because it's the most common default for hidden layers - it's
    # fast to compute and avoids the vanishing gradient problem that
    # sigmoid/tanh run into with deeper networks.
    model.add(layers.Dense(hidden_units, activation="relu", name="Hidden_Layer"))

    # Output layer: softmax converts raw scores into probabilities that sum
    # to 1, which makes sense for a multi-class classification output.
    model.add(layers.Dense(output_units, activation="softmax", name="Output_Layer"))

    return model


def explain_layers(model):
    """Print a short, plain-English breakdown of what each layer is doing."""
    print("\nLayer-by-layer explanation:")
    for layer in model.layers:
        config = layer.get_config()
        units = config.get("units", "N/A")
        activation = config.get("activation", "none")
        print(f"  - {layer.name:15s} | neurons: {str(units):5s} "
              f"| activation: {activation}")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 12 - Practice 2: Simple Neural Network")
    print("=" * 60)

    # Example sizes - imagine 20 input features, 16 hidden neurons,
    # and 4 output classes. These numbers are arbitrary for this practice.
    INPUT_SIZE = 20
    HIDDEN_UNITS = 16
    OUTPUT_UNITS = 4

    try:
        network = build_simple_network(INPUT_SIZE, HIDDEN_UNITS, OUTPUT_UNITS)
    except Exception as error:
        print(f"Something went wrong while building the model: {error}")
        raise

    print("\nModel Summary:")
    network.summary()

    explain_layers(network)

    print("\nDone. Next: experiment with different activation functions "
          "(Practice 3).")