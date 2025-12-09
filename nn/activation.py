from enum import StrEnum

import numpy as np


class ActivationType(StrEnum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"


def activation(
    activation_type: ActivationType, x: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    match activation_type:
        case ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-x))
        case ActivationType.RELU:
            return np.where(x > 0, x, 0)
        case ActivationType.TANH:
            return np.tanh(x)
        case ActivationType.LEAKY_RELU:
            return np.where(x > 0, x, alpha * x)
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")


def derivative(
    activation_type: ActivationType, y: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    match activation_type:
        case ActivationType.SIGMOID:
            return y * (1 - y)

        case ActivationType.RELU:
            return np.where(y > 0, 1, 0)

        case ActivationType.TANH:
            return 1 - y**2

        case ActivationType.LEAKY_RELU:
            return np.where(y > 0, 1, alpha)
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")
