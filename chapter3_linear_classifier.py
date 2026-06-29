"""
CSC-114 — Chapter 3 Demo
A linear classifier in Keras: BUILD -> COMPILE -> FIT -> PREDICT

This recreates the chapter's running example: two clouds of points
(two classes) separated by a single straight line, built the
"real" way (Keras) instead of by hand (Chapter 2's approach).

Backend: defaults to TensorFlow unless KERAS_BACKEND is set
(e.g. os.environ["KERAS_BACKEND"] = "torch" or "jax").
Install with: pip install keras tensorflow numpy matplotlib
"""

import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. Make some fake data: two clouds of 2D points (two classes)
# ----------------------------------------------------------------------
np.random.seed(0)
num_samples_per_class = 1000

# Class 0: points centered around (0, 3)
class_0 = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

# Class 1: points centered around (3, 0)
class_1 = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

inputs = np.vstack([class_0, class_1]).astype("float32")
targets = np.vstack([
    np.zeros((num_samples_per_class, 1), dtype="float32"),
    np.ones((num_samples_per_class, 1), dtype="float32"),
])

# Hold out 20% as validation data — the model NEVER trains on this.
# This is the "honesty check" from the notes (section 6e).
num_val_samples = int(0.2 * len(inputs))
indices = np.random.permutation(len(inputs))
val_idx, train_idx = indices[:num_val_samples], indices[num_val_samples:]

train_inputs, train_targets = inputs[train_idx], targets[train_idx]
val_inputs, val_targets = inputs[val_idx], targets[val_idx]

# ----------------------------------------------------------------------
# 1. BUILD — stack layers into a model
# ----------------------------------------------------------------------
# A single Dense layer with no activation is exactly a linear
# classifier: output = W . x + b, i.e. one straight line in 2D.
# This is the model's "hypothesis space" — it can ONLY ever learn
# a straight-line boundary, nothing curved.
model = keras.Sequential([
    layers.Dense(1, input_shape=(2,))
])

# ----------------------------------------------------------------------
# 2. COMPILE — set the three dials
# ----------------------------------------------------------------------
model.compile(
    optimizer="rmsprop",                 # HOW the model improves
    loss="mean_squared_error",            # WHAT counts as "wrong"
    metrics=["binary_accuracy"],          # HOW you measure success (for you, not training)
)

# ----------------------------------------------------------------------
# 3. FIT — run the training loop
# ----------------------------------------------------------------------
history = model.fit(
    train_inputs, train_targets,
    epochs=30,            # passes over all the training data
    batch_size=16,        # examples per weight update
    validation_data=(val_inputs, val_targets),
)

# ----------------------------------------------------------------------
# 4. PREDICT — use the finished model on new data
# ----------------------------------------------------------------------
predictions = model.predict(val_inputs, batch_size=128)
predicted_classes = (predictions > 0.5).astype(int)

val_accuracy = (predicted_classes == val_targets).mean()
print(f"\nFinal validation accuracy: {val_accuracy:.2%}")

# ----------------------------------------------------------------------
# Bonus: plot the data and the line the model learned
# ----------------------------------------------------------------------
weights, bias = model.layers[0].get_weights()
w1, w2 = weights[:, 0]
b = bias[0]

x_vals = np.linspace(-2, 6, 100)
# The decision boundary is where W.x + b = 0.5 (our threshold)
y_vals = (0.5 - b - w1 * x_vals) / w2

plt.figure(figsize=(6, 6))
plt.scatter(train_inputs[:, 0], train_inputs[:, 1],
            c=train_targets[:, 0], cmap="coolwarm", alpha=0.6, label="train")
plt.plot(x_vals, y_vals, "k--", linewidth=2, label="learned boundary")
plt.title("Linear Classifier — Chapter 3 Demo")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("linear_classifier_result.png", dpi=150)
print("Plot saved to linear_classifier_result.png")
