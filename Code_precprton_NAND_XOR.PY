import numpy as np

# Step function
def step(x):
    return 1 if x >= 0 else 0

# Perceptron training function
def train_perceptron(X, y, lr=0.1, epochs=20, gate_name="Gate"):
    # Add bias input (1)
    X = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X.shape[1])

    print(f"\nTraining {gate_name}\n")

    for epoch in range(1, epochs + 1):
        errors = 0
        print(f"Epoch {epoch}")

        for xi, target in zip(X, y):
            net = np.dot(w, xi)
            y_pred = step(net)
            error = target - y_pred

            if error != 0:
                w = w + lr * error * xi
                errors += 1
                print(f"w = {np.round(w, 3)}")

        print(f"errors this epoch = {errors}")
        print("-" * 30)

        if errors == 0:
            print("Converged")
            return

    print("Did not converge (dataset not linearly separable)")
    print(f"final w = {np.round(w, 3)}")


# ===================== NAND =====================
X_nand = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_nand = np.array([1, 1, 1, 0])

train_perceptron(X_nand, y_nand, gate_name="NAND gate")


# ===================== XOR =====================
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

train_perceptron(X_xor, y_xor, gate_name="XOR gate")
