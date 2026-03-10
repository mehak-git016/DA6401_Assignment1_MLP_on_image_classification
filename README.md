# DA6401 Assignment 1

This repository contains the NumPy-based MLP implementation for DA6401 Assignment 1.
The model supports configurable depth, activation, loss, and optimizers, and is trained on
MNIST (default) and Fashion-MNIST (for Task 2.10).

## Links

- Weights & Biases report: [DA6401 Assign1 W&B]([https://wandb.ai/mehakgupta/DA6401_assign1](https://wandb.ai/ma25m016mehak-indian-institute-of-technology-madras/DA6401_assign1/reports/Assignment-1-Multi-Layer-Perceptron-for-Image-Classification--VmlldzoxNjEzNTcyOQ))
- GitHub repository: [DA6401 Assignment Repo](https://github.com/mehak-git016/DA6401_Assignment1_MLP_on_image_classification)

## Important constraints followed

- Core NN math is implemented manually with NumPy.
- No TensorFlow model API / PyTorch / JAX is used.
- Dataset loading uses `keras.datasets`.
- Supported optimizers: `sgd`, `momentum`, `nag`, `rmsprop`.

## Project layout

```text
da6401_assignment_1/
├── requirements.txt
├── README.md
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── test.py
│   ├── best_model.npy
│   ├── best_config.json
│   ├── ann/
│   │   ├── neural_network.py
│   │   ├── neural_layer.py
│   │   ├── activations.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   ├── utils/
│   │   └── data_loader.py
│   └── wandb_exp_2_*.py
└── ...
```

## Setup

```bash
cd /Users/mehakgupta/Desktop/MTech/sem2/DL/assign1/da6401_assignment_1
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Main scripts

### 1) Train

```bash
python src/train.py
```

Example with explicit args:

```bash
python src/train.py \
  -d mnist \
  -e 10 \
  -b 64 \
  -l mse \
  -o rmsprop \
  -lr 0.00311 \
  -wd 0.0 \
  -nhl 4 \
  -sz 128 64 64 64 \
  -a tanh \
  -wi xavier
```

### 2) Inference

```bash
python src/inference.py --model_path best_model.npy
```

This prints:
- Accuracy
- Precision
- Recall
- F1-score



## Saved artifacts

- `src/best_model.npy`: model weights dictionary (used with `get_weights()` / `set_weights()`).
- `src/best_config.json`: best hyperparameter configuration.
- `src/best_f1.txt`: best test F1 found during sweep flow.


## CLI arguments used (train/inference compatibility)

- `-d, --dataset`: `mnist` or `fashion_mnist`
- `-e, --epochs`
- `-b, --batch_size`
- `-l, --loss`: `mse` or `cross_entropy`
- `-o, --optimizer`: `sgd`, `momentum`, `nag`, `rmsprop`
- `-lr, --learning_rate`
- `-wd, --weight_decay`
- `-nhl, --num_layers` (num_layers is overwritten by dim of hidden_size)
- `-sz, --hidden_size` (list) (num_layers is actually calculated by dim of hidden_size)
- `-a, --activation`: `sigmoid`, `tanh`, `relu`
- `-wi, --weight_init`: `random` or `xavier`
- `-wp, --wandb_project`

## Assignment experiments

The scripts are in `src/wandb_exp_2_1_...py` to `src/wandb_exp_2_10_...py`.


## 2.10 configuration choices

Current 3 configurations used in code:

1. `cfg_1_mnist_best_transfer`: `[128, 64, 64, 64]`, `tanh`, `rmsprop`, `lr=0.003110199715274576`
2. `cfg_2_relu_rmsprop`: `[256, 128, 64]`, `relu`, `rmsprop`, `lr=0.001`
3. `cfg_3_relu_nag`: `[128, 128, 64]`, `relu`, `nag`, `lr=0.0005`

## Notes from implementation

- Output layer returns logits (no softmax at model output).
- `backward()` returns gradients from last layer to first layer.
- Layers expose `grad_W` and `grad_b` after backprop.
- `mnist` is the default dataset in most scripts; Fashion-MNIST is used explicitly for Task 2.10.
