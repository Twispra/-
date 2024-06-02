import numpy as np

w = np.arange(-5, 0, 0.1)
b = np.arange(-5, 1, 0.1)

W, B = np.meshgrid(w, b)
print(W.shape[0])  # 输出 (100, 100)
print(W.shape[1])

print(W)
