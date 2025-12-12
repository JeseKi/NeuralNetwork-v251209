import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from sklearn.datasets import make_moons

from nn.nn import NeuralNetwork, NeuralModule
from nn.activation import ActivationType
from nn.loss import LossType
from nn.train import train, loss

# constants
ENABLE_TRAIN = True  # if false, the model will be loaded from the file, else the model will be trained
ENABLE_VISUALIZATION = True  # if false, the visualization will not be shown, else the visualization will be shown
ENABLE_SAVE_MODEL = False  # if false, the model will not be saved to the file, else the model will be saved to the file

X, y = make_moons(n_samples=10000, noise=0.1, random_state=42)
X_test, y_test = make_moons(n_samples=2000, noise=0.1, random_state=43)
y_train_labels = y.copy()
y_test_labels = y_test.copy()

assert isinstance(X, np.ndarray)
assert isinstance(y, np.ndarray)
assert isinstance(X_test, np.ndarray)
assert isinstance(y_test, np.ndarray)

neural_network = NeuralNetwork(learning_rate=10e-4)
neural_network.add_layer(
    NeuralModule(input_size=2, output_size=4, activation_type=ActivationType.LINEAR)
)
neural_network.add_layer(
    NeuralModule(input_size=4, output_size=4, activation_type=ActivationType.LEAKY_RELU)
)
neural_network.add_layer(
    NeuralModule(input_size=4, output_size=4, activation_type=ActivationType.LEAKY_RELU)
)
neural_network.add_layer(
    NeuralModule(input_size=4, output_size=1, activation_type=ActivationType.SIGMOID)
)

X_list = [X[i].reshape(1, -1) for i in range(X.shape[0])]
y = y.reshape(-1, 1)  # shape(200,) -> shape(200, 1)
y_list = [y[i].reshape(1, -1) for i in range(y.shape[0])]
X_test_list = [X_test[i].reshape(1, -1) for i in range(X_test.shape[0])]
y_test_list = [y_test[i].reshape(1, -1) for i in range(y_test.shape[0])]

if ENABLE_TRAIN:
    train(
        neural_network=neural_network,
        inputs=X_list,
        targets=y_list,
        loss_type=LossType.BCE,
        epochs=30,
        record_interval=1,
    )
else:
    neural_network = NeuralNetwork.load_model("moons.pkl")

total_loss = 0.0
test_pred_probs: list[float] = []
test_pred_labels: list[int] = []
for input_value, target in zip(X_test_list, y_test_list):
    output = neural_network.forward(input_value)
    prob = float(output[-1][0, 0])
    test_pred_probs.append(prob)
    test_pred_labels.append(1 if prob >= 0.5 else 0)
    total_loss += loss(LossType.BCE, output[-1], target)

test_loss_average = total_loss / len(X_test_list)
print(f"test_loss_average: {test_loss_average}")

if ENABLE_SAVE_MODEL:
    neural_network.save_model("moons.pkl")

if ENABLE_VISUALIZATION:
    color_map = {0: "#1f77b4", 1: "#ff7f0e"}
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["训练集真实标签", "测试集真实标签", "测试集预测标签"],
    )

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[int(label)] for label in y_train_labels],
                size=6,
                symbol="circle",
                opacity=0.8,
            ),
            name="Train",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=X_test[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[int(label)] for label in y_test_labels],
                size=7,
                symbol="cross",
                opacity=0.8,
            ),
            name="Test-True",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=X_test[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[int(label)] for label in test_pred_labels],
                size=7,
                symbol="square",
                opacity=0.8,
                line=dict(width=1, color="#111111"),
            ),
            name="Test-Pred",
            text=[f"p={prob:.3f}" for prob in test_pred_probs],
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>%{text}",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title="双半月数据集可视化与测试集预测",
        showlegend=False,
        width=1200,
        height=400,
    )
    fig.update_xaxes(title_text="x1", row=1, col=1)
    fig.update_yaxes(title_text="x2", row=1, col=1)
    fig.update_xaxes(title_text="x1", row=1, col=2)
    fig.update_xaxes(title_text="x1", row=1, col=3)
    fig.update_yaxes(title_text="x2", row=1, col=2)
    fig.update_yaxes(title_text="x2", row=1, col=3)

    fig.show()
