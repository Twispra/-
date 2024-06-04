#随机梯度下降函数，这里面计算loss和gradient都是用的单个值
#性能高，时间复杂度高

import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

loss_list = []

print('Predict (before training', 4, forward(4))

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        print('\tgrad:', x, y, grad)
        l = loss(x, y)

    loss_list.append(l)
    print('Epoch:', epoch, 'w=', w, 'loss=', l)

print('Predict (after training)', 4, forward(4))


plt.plot(range(100), loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()