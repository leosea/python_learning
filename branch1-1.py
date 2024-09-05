# 这是分支1新增的文件1-1

import numpy as np
import matplotlib.pyplot as plt

def heart_shape(x, y):
    return (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0

def plot_heart():
    x = np.linspace(-1.5, 1.5, 300)
    y = np.linspace(-1.5, 1.5, 300)
    X, Y = np.meshgrid(x, y)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, heart_shape(X, Y), levels=[0, 1], colors=['red'])
    plt.axis('equal')
    plt.axis('off')
    plt.title('心形图案')
    plt.show()

# 调用函数绘制心形图案
plot_heart()
