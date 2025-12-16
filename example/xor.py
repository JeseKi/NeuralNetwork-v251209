import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from typing import List

import numpy as np

from nn.nn import NeuralNetwork, NeuralModule
from nn.train import train
from nn.loss import LossType
from nn.activation import ActivationType

INPUT: List[np.ndarray] = [
    np.array([[0, 0]]),
    np.array([[0, 1]]),
    np.array([[1, 0]]),
    np.array([[1, 1]]),
]

OUTPUT: List[np.ndarray] = [
    np.array([[0]]),
    np.array([[1]]),
    np.array([[1]]),
    np.array([[0]]),
]


def visualize_results(
    neural_network: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    grid_size: int = 100,
):
    """
    可视化 XOR 问题的训练结果和决策边界。
    """
    # 1. 生成网格数据用于绘制决策边界
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    decision_x = np.linspace(x_min, x_max, grid_size)
    decision_y = np.linspace(y_min, y_max, grid_size)
    grid_x, grid_y = np.meshgrid(decision_x, decision_y)

    # 构造网格点输入
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # 2. 预测网格点的输出
    probs = np.empty((grid_points.shape[0],))
    for i in range(grid_points.shape[0]):
        # 注意：这里假设 forward 接受 (1, input_size) 的输入
        out = neural_network.forward(grid_points[i].reshape(1, -1))
        # 最后一个模块的输出
        probs[i] = float(out[-1][0, 0])

    decision_z = probs.reshape(grid_x.shape)

    # 3. 绘图
    fig = go.Figure()

    # 添加决策边界的热力图
    fig.add_trace(
        go.Contour(
            x=decision_x,
            y=decision_y,
            z=decision_z,
            colorscale="RdBu",
            zmin=0.0,
            zmax=1.0,
            opacity=0.6,
            line_smoothing=0.85,
            contours=dict(
                coloring="heatmap",
                showlines=False,
            ),
            name="概率分布",
            colorbar=dict(title="输出概率 (Class 1)"),
            hoverinfo="x+y+z",
        )
    )

    # 添加明确的决策边界线 (p=0.5)
    fig.add_trace(
        go.Contour(
            x=decision_x,
            y=decision_y,
            z=decision_z,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0.5,
                coloring="none",
                showlabels=False,
            ),
            line=dict(color="black", width=3, dash="dash"),
            showscale=False,
            name="决策边界 (p=0.5)",
            hoverinfo="skip",
        )
    )

    # 添加原始数据点
    y_flat = y.flatten()

    # Class 0
    fig.add_trace(
        go.Scatter(
            x=X[y_flat == 0][:, 0],
            y=X[y_flat == 0][:, 1],
            mode="markers+text",
            marker=dict(color="blue", size=20, line=dict(width=2, color="white")),
            name="Class 0 (Target 0)",
            text=[f"({p[0]},{p[1]})" for p in X[y_flat == 0]],
            textposition="top center",
        )
    )

    # Class 1
    fig.add_trace(
        go.Scatter(
            x=X[y_flat == 1][:, 0],
            y=X[y_flat == 1][:, 1],
            mode="markers+text",
            marker=dict(color="orange", size=20, line=dict(width=2, color="white")),
            name="Class 1 (Target 1)",
            text=[f"({p[0]},{p[1]})" for p in X[y_flat == 1]],
            textposition="top center",
        )
    )

    fig.update_layout(
        title="XOR 神经网络决策边界可视化",
        xaxis_title="Input x1",
        yaxis_title="Input x2",
        width=700,
        height=600,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
    )

    fig.show()


def visualize_layer_activations(
    neural_network: NeuralNetwork,
    grid_size: int = 100,
):
    """
    可视化隐藏层和输出层的激活函数学习到的内容。
    """
    # 1. 创建网格
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5

    decision_x = np.linspace(x_min, x_max, grid_size)
    decision_y = np.linspace(y_min, y_max, grid_size)
    grid_x, grid_y = np.meshgrid(decision_x, decision_y)

    # 构造网格点
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # 2. 前向传播获取每一层的输出
    outputs_list = [neural_network.forward(p.reshape(1, -1)) for p in grid_points]

    num_layers = len(outputs_list[0])
    all_outputs = []
    for l_idx in range(num_layers):
        # vstack 将 [(1, F), (1, F), ...] 堆叠成 (N, F)
        layer_out = np.vstack([out[l_idx] for out in outputs_list])
        all_outputs.append(layer_out)

    # 3. 绘图
    # 我们有 3 层:
    # L1: Linear (2->2)
    # L2: Sigmoid (2->2)
    # L3: Sigmoid (2->1)

    layer_names = [
        "L1 (Linear)",
        "L2 (Sigmoid)",
        "Output (Sigmoid)",
    ]

    # 计算需要的布局
    rows = len(all_outputs)
    cols = 2  # 前两层有2个神经元，最后一层1个。最大列数为2。

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"{name} - Neuron {i + 1}"
            for name, out in zip(layer_names, all_outputs)
            for i in range(out.shape[1])
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    for row_idx, layer_output in enumerate(all_outputs):
        n_neurons = layer_output.shape[1]

        for col_idx in range(n_neurons):
            activation_map = layer_output[:, col_idx].reshape(grid_x.shape)

            fig.add_trace(
                go.Heatmap(
                    z=activation_map,
                    x=decision_x,
                    y=decision_y,
                    colorscale="Viridis",
                    showscale=False,
                    name=f"L{row_idx + 1}-N{col_idx + 1}",
                ),
                row=row_idx + 1,
                col=col_idx + 1,
            )

    fig.update_layout(
        title="神经网络各层神经元激活值可视化",
        height=300 * rows,
        width=800,
        showlegend=False,
    )
    fig.show()


def main():
    neural_network = NeuralNetwork(learning_rate=10e-2)
    neural_network.add_layer(
        NeuralModule(input_size=2, output_size=2, activation_type=ActivationType.LINEAR)
    )
    neural_network.add_layer(
        NeuralModule(
            input_size=2, output_size=2, activation_type=ActivationType.SIGMOID
        )
    )
    neural_network.add_layer(
        NeuralModule(
            input_size=2, output_size=1, activation_type=ActivationType.SIGMOID
        )
    )

    train(
        neural_network=neural_network,
        inputs=INPUT,
        targets=OUTPUT,
        loss_type=LossType.BCE,
        epochs=10000,
        record_interval=100,
    )

    for input, target in zip(INPUT, OUTPUT):
        output = neural_network.forward(input)
        print(f"Input: {input}, Target: {target}, Output: {output}")

    # 可视化
    print("\nVisualizing results...")
    X_viz = np.concatenate(INPUT, axis=0)  # (4, 2)
    y_viz = np.concatenate(OUTPUT, axis=0)  # (4, 1)
    visualize_results(neural_network, X_viz, y_viz)

    print("Visualizing layer activations...")
    visualize_layer_activations(neural_network)


if __name__ == "__main__":
    main()
