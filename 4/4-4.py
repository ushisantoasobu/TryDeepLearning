# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# 中心差分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)- f(x-h)) / (2*h)

# サンプル式
def function_1(x):
    return 0.01*x**2 + 0.1*x

# サンプル式を描画
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# x = 5 の時の微分を計算、描画
print(numerical_diff(function_1, 5))

# x = 10 の時の微分を計算、描画
print(numerical_diff(function_1, 10))
