# Day 12 — Introduction to Deep Learning & First ANN

**Phase 2 kickoff:** Deep Learning. Today's focus was understanding how
neural networks are structured, what a perceptron actually does, why
activation functions matter, and then putting all of that into practice by
building a real ANN on the Fashion MNIST dataset using TensorFlow/Keras.

---

## 1. What is Deep Learning?

Deep Learning is a subfield of Machine Learning based on artificial neural
networks with multiple layers ("deep" = many hidden layers stacked
together). Instead of manually engineering features the way we did in
classical ML (Linear Regression, K-Means, etc.), a deep learning model
learns the relevant features directly from raw data — pixels, audio
waveforms, raw text — by adjusting millions of internal weights through
training.

## 2. Machine Learning vs Deep Learning

| Aspect | Machine Learning | Deep Learning |
|---|---|---|
| Feature engineering | Mostly manual (we choose the features) | Learned automatically by the network |
| Data requirement | Works well on smaller datasets | Needs larger datasets to perform well |
| Compute requirement | Low to moderate | High (benefits a lot from GPUs) |
| Interpretability | Easier to explain (e.g. regression coefficients) | Harder — often a "black box" |
| Typical algorithms | Linear/Logistic Regression, Decision Trees, K-Means | ANN, CNN, RNN, Transformers |

In short: every Deep Learning model is a Machine Learning model, but not
every ML model is "deep." DL is what you reach for when the data is large,
unstructured (images/audio/text), and the patterns are too complex to
hand-craft features for.

## 3. Applications of Deep Learning

- Image classification & object detection (e.g. YOLO — used it earlier for QueryCam)
- Natural Language Processing (chatbots, translation, summarization)
- Computer Vision (medical imaging, surveillance, self-driving cars)
- Speech recognition and generation
- Recommendation systems

## 4. Artificial Neural Networks (ANN)

An ANN is loosely inspired by how biological neurons connect. It's built
from layers of nodes ("neurons"), where each connection has a **weight**,
and each neuron applies an **activation function** before passing its
output to the next layer.

**Layers in a Neural Network:**
- **Input Layer** — receives the raw data (e.g. pixel values). Doesn't do
  any computation, just defines the shape of what's coming in.
- **Hidden Layer(s)** — where the actual learning happens. Each neuron
  computes a weighted sum of its inputs, adds a bias, and passes the
  result through an activation function.
- **Output Layer** — produces the final prediction (e.g. class
  probabilities for classification).

## 5. What is a Perceptron?

A perceptron is the simplest possible neural unit — a single neuron that
takes inputs, multiplies each by a weight, sums them up with a bias, and
passes the result through an activation function to produce an output.

```
output = activation( (x1*w1 + x2*w2 + ... + xn*wn) + bias )
```

A single perceptron can only learn linearly separable patterns (think of
it as drawing one straight line to separate two classes). Stacking many
perceptrons into layers — which is exactly what an ANN is — lets the
network learn far more complex, non-linear decision boundaries.

## 6. Activation Functions Explored

| Function | Range | Commonly used in |
|---|---|---|
| **ReLU** | `[0, ∞)` | Hidden layers of most modern networks — fast, avoids vanishing gradients |
| **Sigmoid** | `(0, 1)` | Output layer of binary classification problems |
| **Tanh** | `(-1, 1)` | Hidden layers, especially in RNNs — zero-centered, helps optimization |
| **Softmax** | `(0, 1)`, sums to 1 | Output layer of multi-class classification (used in our mini project) |

Without an activation function, stacking layers would be mathematically
pointless — a network with only linear operations collapses into a single
linear function no matter how many layers it has. Activation functions are
what let neural networks approximate non-linear, complex relationships.

`practice3_activation_functions.py` builds the identical architecture
three times (ReLU, Sigmoid, Tanh) to show that the **parameter count stays
the same** — only the math inside each neuron changes.

---

## 7. Mini Project — Fashion MNIST ANN

**Architecture:** `Input(28x28) → Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)`

Dropout was added as a small extra step beyond what was strictly asked —
it randomly "switches off" 20% of hidden neurons during training, which
helps reduce overfitting on the training set.

**Training setup:** Adam optimizer, sparse categorical cross-entropy loss,
10 epochs, batch size 32, 10% of training data held out for validation.

### Final Results

| Metric | Value |
|---|---|
| Final Training Accuracy | **89.24%** |
| Final Validation Accuracy | **88.77%** |
| Test Accuracy | **87.89%** |
| Test Loss | 0.3414 |

Training and validation accuracy stay close to each other across epochs
(see `outputs/accuracy_curve.png`), which means the model isn't overfitting
badly — it's generalizing reasonably well to unseen data.

---

## Folder Structure

```
Day-12/
├── practice1_tensorflow_setup.py     # Verifies TensorFlow/Keras install
├── practice2_simple_nn.py            # Input -> Hidden -> Output ANN + summary
├── practice3_activation_functions.py # Compares ReLU / Sigmoid / Tanh
├── mini_project_fashion_mnist.py     # Full Fashion MNIST ANN pipeline
├── requirements.txt
├── .gitignore
├── README.md
└── outputs/
    ├── dataset_preview.png           # Sample training images
    ├── accuracy_curve.png            # Training vs validation accuracy
    ├── sample_predictions.png        # Predicted vs actual labels
    └── results.txt                   # Final accuracy/loss numbers
```

## How to Run

```powershell
# from inside the Day-12 folder, with the venv activated
python practice1_tensorflow_setup.py
python practice2_simple_nn.py
python practice3_activation_functions.py
python mini_project_fashion_mnist.py
```

`mini_project_fashion_mnist.py` will download the Fashion MNIST dataset
automatically on first run (via `tensorflow.keras.datasets`) and save all
plots into `outputs/`.
