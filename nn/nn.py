from pathlib import Path
from typing import List
import pickle
from enum import StrEnum

import numpy as np
from numpy.random import Generator
from numpy import random

from nn.activation import ActivationType, activation


class InitType(StrEnum):
    XAVIER_UNIFORM = "xavier_uniform"
    HE_UNIFORM = "he_uniform"
    RANDOM_UNIFORM = "random_uniform"


class NeuralModule:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_type: ActivationType,
        seed: int = 42,
        init_type: InitType = InitType.XAVIER_UNIFORM,
    ) -> None:
        """
        Initialize a neural layer.
        Args:
            input_size: The number of input features that each neural receives.
            output_size: The number of output features that the layer produces.
            activation_type: The activation function to use.
            seed: The seed for the random number generator.
            init_type: Weight initialization strategy.
        """
        self.rng: Generator = random.default_rng(seed)
        self.activation_type: ActivationType = activation_type
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.W, self.bias = self._initialize_parameters(init_type)

    def _initialize_parameters(
        self, init_type: InitType
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize weights and bias according to the given strategy.
        Returns:
            A tuple of (weights, bias).
        """
        if init_type == InitType.XAVIER_UNIFORM:
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            W = self.rng.uniform(
                size=(self.input_size, self.output_size), low=-limit, high=limit
            )
        elif init_type == InitType.HE_UNIFORM:
            limit = np.sqrt(6 / self.input_size)
            W = self.rng.uniform(
                size=(self.input_size, self.output_size), low=-limit, high=limit
            )
        else:  # InitType.RANDOM_UNIFORM
            W = self.rng.uniform(
                size=(self.input_size, self.output_size), low=-1.0, high=1.0
            )

        bias = np.zeros((1, self.output_size))
        return W, bias

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

    def save_model(self, path: str | Path) -> None:
        """
        Save the model to a file.
        Args:
            path: The path to save the model.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: str | Path) -> "NeuralNetwork":
        """
        Load the model from a file.
        Args:
            path: The path to load the model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
