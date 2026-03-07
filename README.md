# DA6401 Assignment 1: Multi-Layer Perceptron for Image Classification

## Links

📊 **Weights & Biases Report:** [Add your W&B project link here](https://wandb.ai/your_username/your_project)  
📦 **GitHub Repository:** [Add your GitHub repo link here](https://github.com/your_username/your_repo)

## Overview

This assignment requires building a **configurable neural network from scratch using only NumPy**. The implementation includes:
- Forward and backward propagation
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Loss functions (Cross-Entropy, MSE)
- Optimizers (SGD, Momentum, NAG, RMSProp)
- Training pipeline with mini-batch gradient descent
- Evaluation on MNIST and Fashion-MNIST datasets

**Key Constraint:** Use only **NumPy** for all mathematical operations. PyTorch, TensorFlow, and JAX are prohibited.

---

## Project Structure

```
src/
├── ann/
│   ├── __init__.py
│   ├── activations.py         # ReLU, Sigmoid, Tanh, Softmax + derivatives
│   ├── neural_layer.py        # Dense layer implementation
│   ├── neural_network.py      # Main network orchestration
│   ├── objective_functions.py # Cross-Entropy, MSE losses
│   └── optimizers.py          # SGD, Momentum, NAG, RMSProp
├── utils/
│   ├── __init__.py
│   └── data_loader.py         # Load & preprocess MNIST/Fashion-MNIST
├── train.py                   # Training script with CLI
└── inference.py               # Evaluation & prediction script

models/
└── (saved models directory)

best_model.npy                 # Best trained model weights
best_config.json               # Best hyperparameter configuration
requirements.txt               # Python dependencies
README.md                       # This file
```

---

## Installation & Setup

### 1. Clone/Navigate to Project
```bash
cd /Users/mehakgupta/Desktop/MTech/sem2/DL/assign1/da6401_assignment_1
```

### 2. Use Existing Virtual Environment
```bash
# Use the pre-existing new_venv
source new_venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy>=1.21.0` — All matrix operations and neural network computations
- `scikit-learn>=0.24.2` — Metrics (accuracy, precision, recall, F1, confusion matrix)
- `matplotlib>=3.4.0` — Visualization
- `wandb>=0.12.0` — Experiment tracking
- `tensorflow>=2.10.0` — (Optional) MNIST dataset loading

---

## Workflow Overview

### Step 1: Understanding the Architecture

The network is built layer-by-layer:

```
Input (784 features)
    ↓
Dense Layer (hidden neurons, activation)
    ↓
Dense Layer (hidden neurons, activation)
    ↓
Output Layer (10 classes, linear activation → logits)
    ↓
Loss Function (Cross-Entropy or MSE)
    ↓
Backpropagation & Weight Updates
```

### Step 2: Core Components

#### **activations.py**
Implements forward and derivative computations for:
- **ReLU:** `max(0, x)` — Used in hidden layers
- **Sigmoid:** `1/(1+e^-x)` — Alternative to ReLU
- **Tanh:** `tanh(x)` — Centered sigmoid
- **Softmax:** Converts logits to probabilities (for reference)

#### **neural_layer.py**
Single dense layer that:
- Stores weights `W` and biases `b`
- Computes forward pass: `Z = X @ W + b`, `A = activation(Z)`
- Computes backward pass: returns gradients `dW`, `db`, and upstream gradient `dX`
- Caches `X` and `Z` for backward pass

#### **neural_network.py**
Orchestrates multiple layers:
- Chains layers in forward pass
- Backpropagates gradients in reverse order
- Integrates with optimizers for weight updates
- Saves/loads trained weights

#### **objective_functions.py**
Loss functions:
- **Cross-Entropy:** For multi-class classification with softmax
- **MSE:** Regression or classification alternative

#### **optimizers.py**
Weight update strategies:
- **SGD:** Plain gradient descent
- **Momentum:** Accumulates velocity for faster convergence
- **NAG:** Nesterov look-ahead momentum
- **RMSProp:** Adaptive learning rates per parameter

#### **data_loader.py**
Dataset utilities:
- Loads MNIST (60k training, 10k test) or Fashion-MNIST
- Normalizes pixel values to [0, 1]
- One-hot encodes labels (0-9 → 10-dim vector)
- Creates mini-batches for training

---

## Running the Code

### Training

#### **Basic Training (Default Config)**
```bash
python src/train.py
```

#### **Custom Configuration**
```bash
python src/train.py \
  -d mnist \              # Dataset: 'mnist' or 'fashion_mnist'
  -e 10 \                # Epochs
  -b 64 \                # Batch size
  -lr 0.01 \             # Learning rate
  -o momentum \          # Optimizer: 'sgd', 'momentum', 'nag', 'rmsprop'
  -a relu \              # Activation: 'relu', 'sigmoid', 'tanh'
  -nhl 2 \               # Number of hidden layers
  -sz 128 64 \           # Hidden layer sizes
  -wi xavier \           # Weight init: 'random' or 'xavier'
  -l cross_entropy       # Loss: 'cross_entropy' or 'mse'
```

#### **Example: Train on Fashion-MNIST with 3 Hidden Layers**
```bash
python src/train.py -d fashion_mnist -e 20 -b 32 -lr 0.001 -o rmsprop -a relu -nhl 3 -sz 256 128 64
```

**Output:**
- Prints loss and validation accuracy each epoch
- Saves best model to `best_model.npy`
- Saves config to `best_config.json`

### Inference

#### **Evaluate Saved Model**
```bash
python src/inference.py \
  --model_path best_model.npy \
  -d mnist \             # Must match training dataset
  -sz 128 64 \           # Must match training architecture
  -a relu                # Must match training activation
```

**Output:**
```
==================================================
EVALUATION RESULTS
==================================================
Accuracy:  0.9750
Precision: 0.9745
Recall:    0.9750
F1-Score:  0.9747
==================================================
```

---

## Implementation Details

### Forward Pass
```
For each layer:
  Z = X @ W + b
  A = activation(Z)
```

### Backward Pass
```
Start with loss gradient (from loss function)
For each layer (in reverse):
  dZ = dA * activation_derivative(Z)
  dW = (X.T @ dZ) / batch_size
  db = sum(dZ) / batch_size
  dA_prev = dZ @ W.T
```

### Weight Update (Optimizer)
```
SGD:       W -= lr * dW
Momentum:  v = momentum*v - lr*dW; W += v
NAG:       (look-ahead momentum update)
RMSProp:   adaptive_lr = lr / sqrt(E[dW²] + eps); W -= adaptive_lr * dW
```

---

## Hyperparameter Tuning

### Recommended Starting Points

**For MNIST (easier):**
```bash
-e 10 -b 64 -lr 0.01 -o momentum -a relu -nhl 2 -sz 128 64
```

**For Fashion-MNIST (harder):**
```bash
-e 20 -b 32 -lr 0.001 -o rmsprop -a relu -nhl 3 -sz 256 128 64
```

### Key Tuning Tips
- **Learning Rate:** Start with 0.01, reduce if loss diverges
- **Batch Size:** Larger (128, 256) for faster but noisier updates; smaller (32) for more stable updates
- **Optimizer:** RMSProp often best for images; Momentum good for balanced performance
- **Activation:** ReLU usually best; Sigmoid slower; Tanh alternative
- **Hidden Layers:** 2-3 layers often sufficient; >6 may overfit
- **Neurons per Layer:** 64-256 typically; more for complex tasks

---

## Expected Performance

| Dataset | Optimizer | Accuracy | F1-Score |
|---|---|---|---|
| MNIST | SGD (best) | ~97-98% | ~0.97-0.98 |
| MNIST | Momentum | ~98-99% | ~0.98-0.99 |
| Fashion-MNIST | SGD | ~85-87% | ~0.85-0.87 |
| Fashion-MNIST | RMSProp | ~89-91% | ~0.89-0.91 |

(With proper hyperparameter tuning and sufficient epochs)

---

## Weights & Biases (W&B) Integration

To log experiments to W&B:

```bash
python src/train.py -d mnist -e 10 -b 64 -lr 0.01 -o momentum \
  -wp <your_wandb_project_name>
```

Then view results at: `https://wandb.ai/<username>/<your_wandb_project_name>`

---

## Key Assignment Requirements

✅ **Implemented:**
- [ ] All code uses only NumPy (no auto-differentiation libraries)
- [ ] CLI arguments for full configuration
- [ ] Forward & backward propagation with gradient caching
- [ ] All 4 optimizers (SGD, Momentum, NAG, RMSProp)
- [ ] Multiple activation functions
- [ ] Both loss functions (Cross-Entropy, MSE)
- [ ] Model serialization (`.npy` weights, `.json` config)
- [ ] Inference script with metrics (accuracy, precision, recall, F1)
- [ ] Gradient access: `layer.grad_W`, `layer.grad_b` for autograder
- [ ] Logits output (no softmax in final layer)

---

## Troubleshooting

### Issue: "module not found" errors
**Solution:** Ensure you're running from the project root:
```bash
cd /Users/mehakgupta/Desktop/MTech/sem2/DL/assign1/da6401_assignment_1
python src/train.py ...
```

### Issue: "AttributeError: 'Namespace' object has no attribute..."
**Solution:** Ensure all CLI args match the training config when running inference.

### Issue: NaN values in loss
**Solutions:**
- Reduce learning rate
- Use Xavier weight initialization (`-wi xavier`)
- Increase batch size
- Check data is normalized to [0, 1]

### Issue: Low accuracy after training
**Solutions:**
- Train for more epochs (`-e 20` or more)
- Tune learning rate (try 0.001, 0.005, 0.01)
- Increase hidden layer size
- Use Momentum or RMSProp optimizer
- Add more hidden layers

---

## Files Reference

| File | Purpose |
|---|---|
| `train.py` | Entry point for training; parses CLI args, loads data, runs training loop |
| `inference.py` | Loads saved model and evaluates on test set |
| `activations.py` | Activation functions and their derivatives |
| `neural_layer.py` | Single dense layer with forward/backward |
| `neural_network.py` | Network orchestration (multiple layers) |
| `objective_functions.py` | Cross-Entropy and MSE loss functions |
| `optimizers.py` | SGD, Momentum, NAG, RMSProp implementations |
| `data_loader.py` | MNIST/Fashion-MNIST loading and preprocessing |

---

## Learning Objectives Checklist

- [x] Understand forward and backward propagation *(implemented in neural_layer.py)*
- [x] Implement gradient computation manually *(backward pass computes dW, db)*
- [x] Implement various optimizers *(SGD, Momentum, NAG, RMSProp)*
- [x] Work with activation functions and their derivatives *(ReLU, Sigmoid, Tanh)*
- [x] Train and evaluate neural networks *(train.py + inference.py)*
- [x] Log experiments using Weights & Biases *(wandb integration)*

---

## Contact & Support

For questions about:
- **Assignment requirements:** See DA6401_Assignment_1_2026.pdf
- **Implementation issues:** Debug using `print()` statements in gradient computation
- **Running code:** Ensure you're using `new_venv` with all dependencies installed

Good luck with your implementation!

