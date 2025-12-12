import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from sklearn.datasets import make_moons

from nn.nn import NeuralNetwork, NeuralModule
from nn.activation import ActivationType
from nn.loss import LossType
from nn.train import train, loss


def create_datasets():
    """Create training and test datasets."""
    X, y = make_moons(n_samples=10000, noise=0.1, random_state=42)
    X_test, y_test = make_moons(n_samples=2000, noise=0.1, random_state=43)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    return X, y, X_test, y_test


def create_model():
    """Create and configure the neural network model."""
    neural_network = NeuralNetwork(learning_rate=10e-4)
    neural_network.add_layer(
        NeuralModule(input_size=2, output_size=4, activation_type=ActivationType.LINEAR)
    )
    neural_network.add_layer(
        NeuralModule(
            input_size=4, output_size=4, activation_type=ActivationType.LEAKY_RELU
        )
    )
    neural_network.add_layer(
        NeuralModule(
            input_size=4, output_size=4, activation_type=ActivationType.LEAKY_RELU
        )
    )
    neural_network.add_layer(
        NeuralModule(
            input_size=4, output_size=1, activation_type=ActivationType.SIGMOID
        )
    )

    return neural_network


def train_or_load_model(X_list, y_list, enable_train=True, enable_save_model=False):
    """Either train a new model or load an existing one based on flags."""
    if enable_train:
        neural_network = create_model()
        train(
            neural_network=neural_network,
            inputs=X_list,
            targets=y_list,
            loss_type=LossType.BCE,
            epochs=30,
            record_interval=1,
        )

        if enable_save_model:
            neural_network.save_model("moons.pkl")
    else:
        neural_network = NeuralNetwork.load_model("moons.pkl")

    return neural_network


def evaluate_model(neural_network, X_test_list, y_test_list):
    """Evaluate the model and return test predictions and loss."""
    total_loss = 0.0
    test_pred_probs = []
    test_pred_labels = []

    for input_value, target in zip(X_test_list, y_test_list):
        output = neural_network.forward(input_value)
        prob = float(output[-1][0, 0])
        test_pred_probs.append(prob)
        test_pred_labels.append(1 if prob >= 0.5 else 0)
        total_loss += loss(LossType.BCE, output[-1], target)

    test_loss_average = total_loss / len(X_test_list)
    print(f"test_loss_average: {test_loss_average}")

    return test_pred_probs, test_pred_labels, test_loss_average


def visualize_results(
    neural_network: NeuralNetwork,
    X,
    y,
    X_test,
    y_test,
    test_pred_labels,
    test_pred_probs,
    enable_decision_boundary=True,
    decision_boundary_grid_size=220,
    decision_boundary_margin_ratio=0.08,
):
    """Visualize the training, test, and prediction results."""
    color_map = {0: "#1f77b4", 1: "#ff7f0e"}

    # Generate decision boundary data if enabled
    decision_x = None
    decision_y = None
    decision_z = None

    if enable_decision_boundary:
        X_all = np.concatenate([X, X_test], axis=0)
        x_min, x_max = float(np.min(X_all[:, 0])), float(np.max(X_all[:, 0]))
        y_min, y_max = float(np.min(X_all[:, 1])), float(np.max(X_all[:, 1]))

        x_span = x_max - x_min
        y_span = y_max - y_min
        x_margin = x_span * decision_boundary_margin_ratio
        y_margin = y_span * decision_boundary_margin_ratio

        decision_x = np.linspace(
            x_min - x_margin, x_max + x_margin, decision_boundary_grid_size
        )
        decision_y = np.linspace(
            y_min - y_margin, y_max + y_margin, decision_boundary_grid_size
        )
        grid_x, grid_y = np.meshgrid(decision_x, decision_y)

        # Predict probabilities for each grid point
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        probs = np.empty((grid_points.shape[0],), dtype=np.float64)
        for i in range(grid_points.shape[0]):
            point = grid_points[i].reshape(1, -1)
            output = neural_network.forward(point)
            probs[i] = float(output[-1][0, 0])
        decision_z = probs.reshape(grid_x.shape)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Train-True", "Test-True", "Test-Pred"],
    )

    # Add decision boundary if enabled
    if (
        enable_decision_boundary
        and decision_x is not None
        and decision_y is not None
        and decision_z is not None
    ):
        for col in (1, 2, 3):
            # Probability heatmap
            fig.add_trace(
                go.Contour(
                    x=decision_x,
                    y=decision_y,
                    z=decision_z,
                    colorscale="RdBu",
                    zmin=0.0,
                    zmax=1.0,
                    opacity=0.35,
                    showscale=(col == 3),
                    colorbar=dict(title="p(class=1)") if col == 3 else None,
                    contours=dict(
                        coloring="heatmap",
                        showlines=False,
                    ),
                    hoverinfo="skip",
                    name="Probability Heatmap",
                ),
                row=1,
                col=col,
            )

            # Decision boundary: contour at p=0.5
            fig.add_trace(
                go.Contour(
                    x=decision_x,
                    y=decision_y,
                    z=decision_z,
                    showscale=False,
                    contours=dict(
                        start=0.5,
                        end=0.5,
                        size=0.5,
                        coloring="none",
                        showlabels=False,
                    ),
                    line=dict(color="#111111", width=2),
                    hoverinfo="skip",
                    name="Decision Boundary (p=0.5)",
                ),
                row=1,
                col=col,
            )

    # Training data scatter plot
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[int(label)] for label in y],
                size=6,
                symbol="circle",
                opacity=0.8,
            ),
            name="Train",
        ),
        row=1,
        col=1,
    )

    # Test data scatter plot (true labels)
    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0],
            y=X_test[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[int(label)] for label in y_test],
                size=7,
                symbol="cross",
                opacity=0.8,
            ),
            name="Test-True",
        ),
        row=1,
        col=2,
    )

    # Test data scatter plot (predicted labels)
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
        title="Moons Dataset Visualization and Test Set Prediction",
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


def main():
    """Main function to orchestrate the workflow."""
    # constants
    ENABLE_TRAIN = True  # if false, the model will be loaded from the file, else the model will be trained
    ENABLE_VISUALIZATION = True  # if false, the visualization will not be shown, else the visualization will be shown
    ENABLE_SAVE_MODEL = False  # if false, the model will not be saved to the file, else the model will be saved to the file
    ENABLE_DECISION_BOUNDARY = True  # if true, the decision boundary will be visualized, else the decision boundary will not be visualized
    DECISION_BOUNDARY_GRID_SIZE = 220  # the grid resolution: the larger the finer, but the more forward times (approximately size^2)
    DECISION_BOUNDARY_MARGIN_RATIO = (
        0.08  # the margin ratio of the decision boundary (relative to x/y span)
    )

    # Create datasets
    X, y, X_test, y_test = create_datasets()
    y_train_labels = y.copy()
    y_test_labels = y_test.copy()

    # Prepare data for training
    X_list = [X[i].reshape(1, -1) for i in range(X.shape[0])]
    y_reshaped = y.reshape(-1, 1)  # shape(200,) -> shape(200, 1)
    y_list = [y_reshaped[i].reshape(1, -1) for i in range(y_reshaped.shape[0])]
    X_test_list = [X_test[i].reshape(1, -1) for i in range(X_test.shape[0])]
    y_test_list = [y_test[i].reshape(1, -1) for i in range(y_test.shape[0])]

    # Train or load model
    neural_network = train_or_load_model(
        X_list, y_list, enable_train=ENABLE_TRAIN, enable_save_model=ENABLE_SAVE_MODEL
    )

    # Evaluate the model
    test_pred_probs, test_pred_labels, test_loss_average = evaluate_model(
        neural_network, X_test_list, y_test_list
    )

    # Visualize results if enabled
    if ENABLE_VISUALIZATION:
        visualize_results(
            neural_network,
            X,
            y_train_labels,
            X_test,
            y_test_labels,
            test_pred_labels=test_pred_labels,
            test_pred_probs=test_pred_probs,
            enable_decision_boundary=ENABLE_DECISION_BOUNDARY,
            decision_boundary_grid_size=DECISION_BOUNDARY_GRID_SIZE,
            decision_boundary_margin_ratio=DECISION_BOUNDARY_MARGIN_RATIO,
        )


if __name__ == "__main__":
    main()
