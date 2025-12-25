from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from nn import NeuralModule, NeuralNetwork, ActivationType, LossType
from nn.train import train, loss


def create_datasets() -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    iris: Bunch = load_iris()  # type: ignore
    X: np.ndarray = iris.data  # shape(150, 4)
    y: np.ndarray = iris.target  # shape(150,)
    feature_names = iris.feature_names

    num_classes = 3
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1

    X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(
        X, y_onehot, test_size=0.2, stratify=y, random_state=42
    )

    y_train_labels = np.argmax(y_train_onehot, axis=1)
    y_test_labels = np.argmax(y_test_onehot, axis=1)

    X_train_list = [X_train[i].reshape(1, -1) for i in range(X_train.shape[0])]
    y_train_list = [
        y_train_onehot[i].reshape(1, -1) for i in range(y_train_onehot.shape[0])
    ]
    X_test_list = [X_test[i].reshape(1, -1) for i in range(X_test.shape[0])]
    y_test_list = [
        y_test_onehot[i].reshape(1, -1) for i in range(y_test_onehot.shape[0])
    ]

    return (
        X_train_list,
        y_train_list,
        X_test_list,
        y_test_list,
        X_train,
        y_train_labels,
        X_test,
        y_test_labels,
        feature_names,
    )


def create_model() -> NeuralNetwork:
    neural_network = NeuralNetwork(learning_rate=10e-4)
    neural_network.add_layer(
        NeuralModule(
            input_size=4, output_size=3, activation_type=ActivationType.SIGMOID
        )
    )
    return neural_network


def train_or_load_model(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    enable_train=True,
    enable_save_model=False,
):
    """Either train a new model or load an existing one based on flags."""
    if enable_train:
        neural_network = create_model()
        train(
            neural_network=neural_network,
            inputs=X_list,
            targets=y_list,
            loss_type=LossType.BCE,
            epochs=1000,
            record_interval=10,
        )

        if enable_save_model:
            neural_network.save_model("iris.pkl")
    else:
        neural_network = NeuralNetwork.load_model("iris.pkl")

    return neural_network


def evaluate_model(
    neural_network: NeuralNetwork,
    X_test_list: List[np.ndarray],
    y_test_list: List[np.ndarray],
):
    """Evaluate the model and return test predictions and loss."""
    total_loss = 0.0
    test_pred_probs = []
    test_pred_labels = []
    test_true_labels = []

    for input_value, target in zip(X_test_list, y_test_list):
        output = neural_network.forward(input_value)
        probs = output[-1][0]
        pred_label = int(np.argmax(probs))
        true_label = int(np.argmax(target[0]))

        test_pred_probs.append(probs.tolist())
        test_pred_labels.append(pred_label)
        test_true_labels.append(true_label)
        total_loss += loss(LossType.BCE, output[-1], target)

    test_loss_average = total_loss / len(X_test_list)

    correct = sum(
        [1 for pred, true in zip(test_pred_labels, test_true_labels) if pred == true]
    )
    accuracy = correct / len(test_pred_labels)

    print(f"test_loss_average: {test_loss_average:.4f}")
    print(f"test_accuracy: {accuracy:.2%} ({correct}/{len(test_pred_labels)})")

    return (
        test_pred_probs,
        test_pred_labels,
        test_true_labels,
        test_loss_average,
        accuracy,
    )


def visualize_results(
    neural_network: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_pred_labels: List[int],
    feature_names: List[str],
    f1_idx: int = 2,
    f2_idx: int = 3,
    grid_size: int = 150,
):
    color_map = {0: "#636EFA", 1: "#EF553B", 2: "#00CC96"}
    class_names = ["Setosa", "Versicolor", "Virginica"]

    x_min, x_max = X_train[:, f1_idx].min() - 0.5, X_train[:, f1_idx].max() + 0.5
    y_min, y_max = X_train[:, f2_idx].min() - 0.5, X_train[:, f2_idx].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )

    X_grid = np.tile(np.mean(X_train, axis=0), (xx.size, 1))
    X_grid[:, f1_idx] = xx.ravel()
    X_grid[:, f2_idx] = yy.ravel()

    Z = np.zeros(xx.size)
    for i in range(X_grid.shape[0]):
        output = neural_network.forward(X_grid[i].reshape(1, -1))
        Z[i] = np.argmax(output[-1])  # type: ignore
    Z = Z.reshape(xx.shape)  # type: ignore

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Train-True", "Test-True", "Test-Pred"],
        horizontal_spacing=0.05,
    )

    for i, (X_data, y_data, title) in enumerate(
        [
            (X_train, y_train, "Train-True"),
            (X_test, y_test, "Test-True"),
            (X_test, np.array(test_pred_labels), "Test-Pred"),
        ]
    ):
        col = i + 1

        fig.add_trace(
            go.Contour(
                x=np.linspace(x_min, x_max, grid_size),
                y=np.linspace(y_min, y_max, grid_size),
                z=Z,
                opacity=0.3,
                showscale=False,
                colorscale=[[0, color_map[0]], [0.5, color_map[1]], [1, color_map[2]]],
                zmin=0,
                zmax=2,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

        for class_idx in range(3):
            mask = y_data == class_idx
            fig.add_trace(
                go.Scatter(
                    x=X_data[mask, f1_idx],
                    y=X_data[mask, f2_idx],
                    mode="markers",
                    name=class_names[class_idx],
                    marker=dict(
                        color=color_map[class_idx],
                        size=8,
                        line=dict(width=1, color="White"),
                    ),
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

        fig.update_xaxes(title_text=feature_names[f1_idx], row=1, col=col)
        fig.update_yaxes(title_text=feature_names[f2_idx], row=1, col=col)

    fig.update_layout(
        title=f"Iris dataset visualization (feature: {feature_names[f1_idx]} vs {feature_names[f2_idx]})",
        width=1200,
        height=500,
        template="plotly_white",
    )
    fig.write_html("iris_vis.html")
    print("Visualization results saved to iris_vis.html")


def visualize_layer_activations(
    neural_network: NeuralNetwork,
    X_train: np.ndarray,
    feature_names: List[str],
    f1_idx: int = 2,
    f2_idx: int = 3,
    grid_size: int = 100,
):
    x_min, x_max = X_train[:, f1_idx].min() - 0.5, X_train[:, f1_idx].max() + 0.5
    y_min, y_max = X_train[:, f2_idx].min() - 0.5, X_train[:, f2_idx].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )
    X_grid = np.tile(np.mean(X_train, axis=0), (xx.size, 1))
    X_grid[:, f1_idx] = xx.ravel()
    X_grid[:, f2_idx] = yy.ravel()

    all_layer_outputs = []
    for i in range(X_grid.shape[0]):
        all_layer_outputs.append(neural_network.forward(X_grid[i].reshape(1, -1)))

    num_layers = len(all_layer_outputs[0])

    processed_outputs = []
    for l_idx in range(num_layers):
        layer_data = np.vstack([out[l_idx] for out in all_layer_outputs])
        processed_outputs.append(layer_data)

    rows = num_layers
    cols = max(out.shape[1] for out in processed_outputs)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"Layer {l_idx + 1} - Neuron {n_idx + 1}"
            for l_idx in range(rows)
            for n_idx in range(cols)
        ],
        vertical_spacing=0.1,
    )

    for l_idx, layer_out in enumerate(processed_outputs):
        n_neurons = layer_out.shape[1]
        for n_idx in range(n_neurons):
            z = layer_out[:, n_idx].reshape(xx.shape)
            fig.add_trace(
                go.Heatmap(
                    x=np.linspace(x_min, x_max, grid_size),
                    y=np.linspace(y_min, y_max, grid_size),
                    z=z,
                    colorscale="Viridis",
                    showscale=False,
                ),
                row=l_idx + 1,
                col=n_idx + 1,
            )

    fig.update_layout(
        title="Neuron activation visualization: observe how features are extracted layer by layer",
        height=300 * rows,
        width=250 * cols,
        template="plotly_white",
    )
    fig.write_html("iris_activations.html")
    print("Neuron activation visualization results saved to iris_activations.html")


def main():
    (
        X_train_list,
        y_train_list,
        X_test_list,
        y_test_list,
        X_train,
        y_train_labels,
        X_test,
        y_test_labels,
        feature_names,
    ) = create_datasets()

    neural_network = train_or_load_model(X_train_list, y_train_list)

    test_pred_probs, test_pred_labels, test_true_labels, test_loss_average, accuracy = (
        evaluate_model(neural_network, X_test_list, y_test_list)
    )

    print("\ntest_pred_details:")
    print(f"test_pred_labels: {test_pred_labels}")
    print(f"test_true_labels: {test_true_labels}")

    visualize_results(
        neural_network,
        X_train,
        y_train_labels,
        X_test,
        y_test_labels,
        test_pred_labels,
        feature_names,
    )

    visualize_layer_activations(neural_network, X_train, feature_names)


if __name__ == "__main__":
    main()
