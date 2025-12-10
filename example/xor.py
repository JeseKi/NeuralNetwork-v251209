from typing import List

import numpy as np

from nn.nn import NeuralNetwork, NeuralModule
from nn.train import train
from nn.loss import LossType
from nn.activation import ActivationType

INPUT: List[np.ndarray] = [
    np.array([[0, 0]]),
    np.array([[0, 1]]),
    np.array([[1, 0]]),
    np.array([[1, 1]]),
]

OUTPUT: List[np.ndarray] = [
    np.array([[0]]),
    np.array([[1]]),
    np.array([[1]]),
    np.array([[0]]),
]

neural_network = NeuralNetwork(learning_rate=10e-2)
neural_network.add_layer(
    NeuralModule(input_size=2, output_size=2, activation_type=ActivationType.LINEAR)
)
neural_network.add_layer(
    NeuralModule(input_size=2, output_size=2, activation_type=ActivationType.SIGMOID)
)
neural_network.add_layer(
    NeuralModule(input_size=2, output_size=1, activation_type=ActivationType.SIGMOID)
)

train(
    neural_network=neural_network,
    inputs=INPUT,
    targets=OUTPUT,
    loss_type=LossType.MSE,
    epochs=10000,
    record_interval=100,
)

for input, target in zip(INPUT, OUTPUT):
    output = neural_network.forward(input)
    print(f"Input: {input}, Target: {target}, Output: {output}")
