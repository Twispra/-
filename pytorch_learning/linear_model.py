import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
#前馈，y = x * w,这里的y是预测值
def forward(x):
    return x * w
#损失函数loss = (y预测值 - y)^2
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []#记录多个权重
mse_list = []#w权重对应的损失值

#从0.0到4.0，间隔为0.1
for w in np.arange(0.0, 4.1, 0.1):
    print('w=',w)
    l_sum = 0
    #合并x，y的值
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l_sum += loss(x_val, y_val)
        print('\t',x_val, y_val, y_pred_val, loss(x_val, y_val))
    print('MSE=',l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('LOSS')
plt.xlabel('w')
plt.show()