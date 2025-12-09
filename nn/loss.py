from enum import StrEnum

import numpy as np


class LossType(StrEnum):
    MSE = "MSE"
    BCE = "BCE"


def loss(loss_type: LossType, y_pred: np.ndarray, y: np.ndarray) -> float:
    match loss_type:
        case LossType.MSE:
            return float(np.mean(0.5 * (y - y_pred) ** 2))

        case LossType.BCE:
            return float(np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)))

        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")


def grad(loss_type: LossType, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    N = y_pred.shape[0]
    match loss_type:
        case LossType.MSE:
            return (y_pred - y) / N

        case LossType.BCE:
            return (y_pred - y) / (y_pred * (1 - y_pred)) / N

        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")
