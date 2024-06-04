#如果cost function是一个不规则有多个极小值的函数(非凸函数)，分治发就不一定能得到最佳答案
#这时就要使用梯度下降算法，对cost function中cost求导数，可以看出增减性
#也是只能找到局部最优


#!!!!但是这种用全部值求出来的梯度下降函数遇到鞍点会停止，因为导数为0了嘛
#所以要用Stochastic Gradient descent随机梯度下降
#性能低下，时间复杂度低

import matplotlib.pyplot as plt

#prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 6.0, 9.0]
learning_rate = 0.01
#Initial guess of weight
w = 1.0
#Define the model:Linear Model.   calculator the y_hat
def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_hat = forward(x)
        cost += (y_hat - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

epochs = []
losses = []

print('Predict (before training', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= learning_rate * grad_val
    epochs.append(epoch)
    losses.append(cost_val)
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)

print('Predict (after training)', 4, forward(4))

#绘制图表
plt.plot(epochs, losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()