
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

For each model variant, the script checks if a trained model already exists. If not, it trains a new model; otherwise, it loads the existing model and displays the results.

**To force train new models, please delete existing ones (*.pth).**

## Results

The training and validation accuracy, along with the loss for each epoch, are logged to TensorBoard. You can visualize the results by running:

```bash
tensorboard --logdir=runs
```

### Training vs Test Accuracy
The training process involves multiple steps:

1. **Model Initialization**: The model, optimizer, and learning rate scheduler are initialized.
2. **TensorBoard Logging**: A `SummaryWriter` is created for TensorBoard logging, including the model graph.
3. **Training Loop**: For each epoch:
   - The model is set to training mode.
   - A progress bar is displayed using `tqdm`.
   - For each batch, the model's predictions are computed, the loss is calculated, and backpropagation is performed.
   - Training accuracy and loss are logged to TensorBoard.
4. **Validation**: At the end of each epoch, the model is evaluated on the validation set. Validation accuracy and loss are logged.
5. **Learning Rate Adjustment**: The scheduler adjusts the learning rate based on validation loss.
6. **Model Checkpointing**: The model is saved if validation loss improves.
7. **Early Stopping**: Training stops early if the learning rate drops below a threshold.

**NOTE** - We have resized input to 32X32 to adapt to the original Lenet5 network proposed in papers.


Red vertical lines show where the best model checkpoint was taken (Based on the **validation** set)
#### LeNet5 Vanilla
This model represents the basic implementation of LeNet5 without any additional modifications. The graph below shows the comparison between training and test accuracy.

![Lenet5_vanilla.png](assets\Lenet5_vanilla.png "Lenet5_vanilla.png")

#### LeNet5 with Dropout
In this variant, dropout layer was added to prevent overfitting. The dropout randomly zeroes out neorons before the linear layers, thus preventing the network from becoming too dependent on specific neurons and forces it to learn redundant representations of features. It is apparent the train accuracy less overfitted to the dataset from the vanilla version, as differences between train and test acucuracies are lower. Moreover, this trial had a better test accuracy that the vanilla.


![Lenet5_dropout.png](assets\Lenet5_dropout.png "Lenet5_dropout.png")

#### LeNet5 with Batch Normalization
This version of LeNet5 incorporates batch normalization layers. This trial was the fastest among all, achieving the best model in the earliest epoch and maintaining the highest learning rate the longest. This outcome aligns with expectations since batch normalization reduces internal covariate shift by normalizing inputs and aids optimization by keeping gradients in a consistent range, as explained by Sergey Ioffe et al. and as outlined in the exercise.

This trial had the best results concerning accuracy.

![Lenet5_bn.png](assets\Lenet5_bn.png "Lenet5_bn.png")

#### LeNet5 with Weight Decay
This model uses L2 regularization (weight decay) to penalize large weights and reduce overfitting. The regularization term reduces the risk of capturing noise and fluctuations in the training data as if they were meaningful patterns by penalizing these large values, thus enforcing the model to optimize better for the general case.
In terms of overfitting - this trial outperformed, reaching the minimal difference between test and train accuracies.

![LeNet5_weight_decay.png](assets\LeNet5_weight_decay.png "LeNet5_weight_decay.png")


| Model                       | Train Accuracy | Test Accuracy |
|-----------------------------|----------------|---------------|
| LeNet5 Vanilla              | 93.71%         | 89.15%        |
| LeNet5 with Dropout         | 92.21%         | 89.96%        |
| LeNet5 with Batch Normalization | 95.57%     | 91.35%        |
| LeNet5 with Weight Decay    | 90.80%         | 88.45%        |

## Acknowledgements

This project was inspired by the original LeNet-5 model proposed by Yann LeCun et al. for handwritten digit recognition. The Fashion MNIST dataset used in this project is a collection of Zalando's article images, serving as a drop-in replacement for the original MNIST dataset.

The implementation also incorporates Batch Normalization as proposed by Sergey Ioffe and Christian Szegedy in their paper ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167).

### References

- [LeNet-5: Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/lenet/)
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)

