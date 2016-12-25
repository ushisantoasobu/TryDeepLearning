# coding: utf-8
import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状の配列を作成

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 元の値に戻す

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
# print(gradient_descent(function_2, init_x=init_x, lr=1.0, step_num=100)) # 学習率1.0だと当たり前だが何も変わらない
# print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
# print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)) # 学習率が小さい = ほとんど変化ない（これも当たり前）
