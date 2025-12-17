from enum import StrEnum

import numpy as np


class ActivationType(StrEnum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"


def activation(
    activation_type: ActivationType, x: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    match activation_type:
        case ActivationType.LINEAR:
            return x
        case ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-x))
        case ActivationType.RELU:
            return np.where(x > 0, x, 0)
        case ActivationType.TANH:
            return np.tanh(x)
        case ActivationType.LEAKY_RELU:
            return np.where(x > 0, x, alpha * x)
        case ActivationType.SOFTMAX:
            x_shifted = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")


def derivative(
    activation_type: ActivationType, y: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    match activation_type:
        case ActivationType.LINEAR:
            return np.ones_like(y)

        case ActivationType.SIGMOID:
            return y * (1 - y)

        case ActivationType.RELU:
            return np.where(y > 0, 1, 0)

        case ActivationType.TANH:
            return 1 - y**2

        case ActivationType.LEAKY_RELU:
            return np.where(y > 0, 1, alpha)
        case ActivationType.SOFTMAX:
            # Softmax 的导数是其 Jacobian 矩阵
            # 对于向量化计算，这里返回的是用于元素级乘法的形式
            # 注意：当与 CCE 损失函数结合时，梯度会简化为 (y_pred - y)
            # 这里提供通用的导数形式：y * (1 - y) for diagonal, -y_i * y_j for off-diagonal
            # 但在实际反向传播中，通常直接在损失函数中处理
            return y * (1 - y)  # 简化形式，仅用于对角元素
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")
