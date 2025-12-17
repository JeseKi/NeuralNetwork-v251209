from enum import StrEnum

import numpy as np


class LossType(StrEnum):
    MSE = "MSE"
    BCE = "BCE"
    CCE = "CCE"  # Categorical Cross-Entropy，用于多分类任务


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

        case LossType.CCE:
            # Categorical Cross-Entropy
            # y: one-hot 编码的真实标签，形状为 (batch_size, num_classes)
            # y_pred: softmax 输出的预测概率，形状为 (batch_size, num_classes)
            y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)
            # 只计算真实类别的交叉熵：-sum(y * log(y_pred))
            return float(-np.mean(np.sum(y * np.log(y_pred_clipped), axis=-1)))

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

        case LossType.CCE:
            # CCE 对 softmax 输出的梯度
            # 当 softmax 与 CCE 结合时，梯度简化为：(y_pred - y) / N
            # 这是因为 d(CCE)/d(z) = d(CCE)/d(softmax) * d(softmax)/d(z) = y_pred - y
            return (y_pred - y) / N

        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")
