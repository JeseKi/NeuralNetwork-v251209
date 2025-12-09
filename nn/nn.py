from typing import List

import numpy as np
from numpy.random import Generator
from numpy import random

from nn.activation import ActivationType, activation


class NeuralModule:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_type: ActivationType,
        seed: int = 42,
    ) -> None:
        """
        Initialize a neural layer.
        Args:
            input_size: The number of input features that each neural receives.
            output_size: The number of output features that the layer produces.
            activation_type: The activation function to use.
            seed: The seed for the random number generator.
        """
        self.rng: Generator = random.default_rng(seed)
        self.activation_type: ActivationType = activation_type
        self.input_size: int = input_size
        self.output_size: int = output_size

        self.W: np.ndarray = self.rng.uniform(
            size=(input_size, output_size), low=-1, high=1
        )
        self.bias: np.ndarray = self.rng.uniform(size=(1, output_size), low=-1, high=1)

    def forward(self, input: np.ndarray) -> np.ndarray:
        return activation(self.activation_type, input @ self.W + self.bias)


class NeuralNetwork:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learn_rate: float = learning_rate

        self.layers: List[NeuralModule] = []

    def forward(self, input: np.ndarray) -> List[np.ndarray]:
        """
        Returns the output of the neural network.

        Args:
            input: The input to the neural network.
        Returns:
            The output of the neural network. A list of arrays, each representing the output of a layer.
        """
        output: List[np.ndarray] = []
        last_output: np.ndarray = input.copy()
        for layer in self.layers:
            layer_output = layer.forward(last_output)
            output.append(layer_output)
            last_output = layer_output
        return output

    def add_layer(self, layer: NeuralModule) -> None:
        self.layers.append(layer)
