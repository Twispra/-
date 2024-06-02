import numpy as np
import matplotlib.pyplot as plt


# 设定函数为 y = 2x + 1
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([3.0, 5.0, 7.0])

# 预测函数
def forward(x, w, b):
    return x * w + b

# 损失函数
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# 定义 w 和 b 的范围
w = np.arange(-5, 5, 0.1)
b = np.arange(-5, 5, 0.1)

# 创建网格
W, B = np.meshgrid(w, b)

# 计算损失值
L = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val, W[i, j], B[i, j])
        L[i, j] = l_sum / len(x_data)

# 绘制 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, L, cmap='viridis')

# 添加标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface')

plt.show()
