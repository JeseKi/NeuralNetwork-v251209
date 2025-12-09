from typing import List

from nn.nn import NeuralNetwork
from nn.loss import LossType, loss, grad
from nn.activation import derivative

from tqdm import tqdm
import numpy as np


def backward(
    neural_network: NeuralNetwork,
    input: np.ndarray,
    target: np.ndarray,
    output: List[np.ndarray],
    loss_type: LossType = LossType.MSE,
) -> float:
    _loss = loss(loss_type, output[-1], target)
    grad_loss = grad(loss_type=loss_type, y_pred=output[-1], y=target)
    grad_last_layer = grad_loss
    last_layer_W: np.ndarray | None = None
    reversed_current_layer_index = 0

    for layer, h in zip(reversed(neural_network.layers), reversed(output)):
        grad_h = derivative(
            neural_network.layers[-reversed_current_layer_index].activation_type, h
        )

        # output layer delta
        if reversed_current_layer_index == 0:
            grad_current_layer = grad_loss * grad_h
        # hidden layer delta
        else:
            assert last_layer_W is not None, "Last layer weight is not set"
            grad_current_layer = (grad_last_layer @ last_layer_W.T) * grad_h

        grad_last_layer = grad_current_layer
        # weight delta
        if reversed_current_layer_index + 1== len(neural_network.layers):
            _input = input.copy()
        else:
            _input = output[-(reversed_current_layer_index + 2)]

        # gradient of weight
        grad_W = _input.T @ grad_current_layer
        grad_bias = grad_current_layer
        last_layer_W = layer.W.copy()
        layer.W -= neural_network.learn_rate * grad_W
        layer.bias -= neural_network.learn_rate * grad_bias
        reversed_current_layer_index += 1

    return _loss


def train(
    neural_network: NeuralNetwork,
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    loss_type: LossType = LossType.MSE,
    epochs: int = 1000,
    record_interval: int = 100,
) -> None:
    for epoch in tqdm(range(epochs)):
        total_loss: float = 0.0
        for input, target in zip(inputs, targets):
            output: List[np.ndarray] = neural_network.forward(input)
            _loss: float = backward(neural_network, input, target, output, loss_type)
            total_loss += _loss
        print(
            f"Epoch {epoch} - Loss: {total_loss / len(inputs)}"
        ) if epoch % record_interval == 0 else None
