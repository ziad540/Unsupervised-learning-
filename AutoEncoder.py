import numpy as np
# Activation Functions and their Derivatives
def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return sigmoid(z) * (1 - sigmoid(z))

def tanh(z): return np.tanh(z)
def tanh_deriv(z): return 1 - np.tanh(z) ** 2

class Autoencoder:
    def __init__(self, layer_dims, activation="relu", lr=0.001, l2=1e-4):
        self.layer_dims = layer_dims
        self.lr = lr
        self.l2 = l2

        self.act, self.act_deriv = {
            "relu": (relu, relu_deriv),
            "sigmoid": (sigmoid, sigmoid_deriv),
            "tanh": (tanh, tanh_deriv)
        }[activation]

        self._init_weights()

    def _init_weights(self):
        self.W, self.b = [], []
        for i in range(len(self.layer_dims)-1):
            self.W.append(
                np.random.randn(self.layer_dims[i], self.layer_dims[i+1]) * 0.01 # He initialization
            )
            self.b.append(np.zeros((1, self.layer_dims[i+1]))) # Bias initialization

    def forward(self, X):
        self.Z, self.A = [], [X] # Z: pre-activation, A: activation
        for W, b in zip(self.W, self.b): 
            z = self.A[-1] @ W + b # Linear step
            self.Z.append(z) # Store pre-activation
            self.A.append(self.act(z)) # Activation step
        return self.A[-1] # Output

    def backward(self, X): 
        m = X.shape[0] # Number of samples
        dA = (self.A[-1] - X) / m # Derivative of MSE loss

        for i in reversed(range(len(self.W))): # reversed meaning from last layer to first
            dZ = dA * self.act_deriv(self.Z[i]) # Derivative of activation
            dW = self.A[i].T @ dZ + self.l2 * self.W[i] # Derivative of weights with L2 regularization
            db = np.sum(dZ, axis=0, keepdims=True) # Derivative of biases

            dA = dZ @ self.W[i].T # Derivative for next layer

            self.W[i] -= self.lr * dW # Update weights
            self.b[i] -= self.lr * db

    def train(self, X, epochs=100, batch_size=32, decay=0.99):
        losses = []
        for epoch in range(epochs):
            idx = np.random.permutation(len(X)) # Shuffle data
            X_shuffled = X[idx]

            for i in range(0, len(X), batch_size): 
                batch = X_shuffled[i:i+batch_size] # Mini-batch
                out = self.forward(batch) # Forward pass
                self.backward(batch) # Backward pass

            loss = np.mean((X - self.forward(X)) ** 2) # Compute MSE loss
            losses.append(loss)
            self.lr *= decay # Decay learning rate

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, MSE={loss:.4f}") # Print progress

        return losses

    def encode(self, X): 
        A = X # Input
        for i in range(len(self.W)//2): # Encode only up to the bottleneck
            A = self.act(A @ self.W[i] + self.b[i]) # Activation step
        return A
