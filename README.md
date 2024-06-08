
# Fashion MNIST with LeNet-5

This project implements and evaluates different variants of the LeNet-5 model on the Fashion MNIST dataset. It was done as homework for the Deep Learning course (0510725502) at Tel Aviv University.

The implemented variants include:

1. **Vanilla LeNet-5**
2. **LeNet-5 with Batch Normalization**
3. **LeNet-5 with Dropout**
4. **LeNet-5 with Weight Decay**

Each variant is trained and evaluated using PyTorch, with the results logged to TensorBoard for easy visualization.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Variants](#model-variants)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation

Checkout to the project directory and install the dependencies:

```bash
pip install -r requirements.txt
```


## Usage

To train and evaluate the models, run the following command:

```bash
fashion_mnist_lenet5.ipynb
```

This will train all the variants of the LeNet-5 model and log the results to TensorBoard and save best models locally.

## Model Variants

### 1. Vanilla LeNet-5
The basic LeNet-5 model without any additional modifications.

### 2. LeNet-5 with Batch Normalization
This variant includes Batch Normalization layers after each convolutional layer to stabilize and speed up the training process.

### 3. LeNet-5 with Dropout
In this variant, Dropout is applied after the first fully connected layer to prevent overfitting.

### 4. LeNet-5 with Weight Decay
This model uses L2 regularization (weight decay) to penalize large weights and reduce overfitting.

## Results

The training and validation accuracy, along with the loss for each epoch, are logged to TensorBoard. You can visualize the results by running:

```bash
tensorboard --logdir=runs
```

## Acknowledgements

This project was inspired by the original LeNet-5 model proposed by Yann LeCun et al. for handwritten digit recognition. The Fashion MNIST dataset used in this project is a collection of Zalando's article images, serving as a drop-in replacement for the original MNIST dataset.
