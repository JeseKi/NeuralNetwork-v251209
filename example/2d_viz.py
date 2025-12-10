import numpy as np
from sklearn.datasets import make_moons

from nn.nn import NeuralNetwork, NeuralModule
from nn.activation import ActivationType
from nn.loss import LossType
from nn.train import train, loss

X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_test, y_test = make_moons(n_samples=100, noise=0.1, random_state=43)

assert isinstance(X, np.ndarray)
assert isinstance(y, np.ndarray)
assert isinstance(X_test, np.ndarray)
assert isinstance(y_test, np.ndarray)

neural_network = NeuralNetwork(learning_rate=10e-4)
neural_network.add_layer(
    NeuralModule(input_size=2, output_size=2, activation_type=ActivationType.LINEAR)
)
neural_network.add_layer(
    NeuralModule(input_size=2, output_size=4, activation_type=ActivationType.SIGMOID)
)
neural_network.add_layer(
    NeuralModule(input_size=4, output_size=8, activation_type=ActivationType.SIGMOID)
)
neural_network.add_layer(
    NeuralModule(input_size=8, output_size=8, activation_type=ActivationType.SIGMOID)
)
neural_network.add_layer(
    NeuralModule(input_size=8, output_size=2, activation_type=ActivationType.SIGMOID)
)
neural_network.add_layer(
    NeuralModule(input_size=2, output_size=1, activation_type=ActivationType.SIGMOID)
)

X_list = [X[i].reshape(1, -1) for i in range(X.shape[0])]
y = y.reshape(-1, 1)  # shape(200,) -> shape(200, 1)
y_list = [y[i].reshape(1, -1) for i in range(y.shape[0])]
X_test_list = [X_test[i].reshape(1, -1) for i in range(X_test.shape[0])]
y_test_list = [y_test[i].reshape(1, -1) for i in range(y_test.shape[0])]

train(
    neural_network=neural_network,
    inputs=X_list,
    targets=y_list,
    loss_type=LossType.MSE,
    epochs=2900,
    record_interval=100,
)

total_loss = 0.0
for input_value, target in zip(X_test_list, y_test_list):
    output = neural_network.forward(input_value)
    total_loss += loss(LossType.MSE, output[-1], target)

test_loss_average = total_loss / len(X_test_list)
print(f"test_loss_average: {test_loss_average}")

neural_network.save_model("neural_network.pkl")