from enum import StrEnum

import numpy as np


class LossType(StrEnum):
    MSE = "MSE"
    BCE = "BCE"


EPSILON = 1e-12


def loss(loss_type: LossType, y_pred: np.ndarray, y: np.ndarray) -> float:
    match loss_type:
        case LossType.MSE:
            return float(np.mean(0.5 * (y - y_pred) ** 2))

        case LossType.BCE:
            y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)
            return float(
                np.mean(
                    -y * np.log(y_pred_clipped) - (1 - y) * np.log(1 - y_pred_clipped)
                )
            )

        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")


def grad(loss_type: LossType, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    N = y_pred.shape[0]
    match loss_type:
        case LossType.MSE:
            return (y_pred - y) / N

        case LossType.BCE:
            y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)
            return (y_pred_clipped - y) / (y_pred_clipped * (1 - y_pred_clipped)) / N

        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")
