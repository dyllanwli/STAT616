import pandas as pd
import numpy as np

def read_data(x, y, p="./data/normalized.csv"):
    df = pd.read_csv(p)
    return df[x], df[y]

def da(y,y_p,x):
    return (y-y_p)*(-x)

def db(y,y_p):
    return (y-y_p)*(-1)

def calc_loss(a,b,x,y):
    tmp = y - (a * x + b)
    tmp = tmp ** 2  # 对矩阵内的每一个元素平方
    SSE = sum(tmp) / (2 * len(x))
    return SSE

def draw_hill(x, y):
    a = np.linspace(-20, 20, 100)
    print(a)
    b = np.linspace(-20, 20, 100)
    x = np.array(x)
    y = np.array(y)

    allSSE = np.zeros(shape=(len(a),len(b)))
    for ai in range(0,len(a)):
        for bi in range(0,len(b)):
            a0 = a[ai]
            b0 = b[bi]
            SSE = calc_loss(a=a0, b=b0, x=x, y=y)
            allSSE[ai][bi] = SSE

    a,b = np.meshgrid(a, b)

    return [a, b, allSSE]