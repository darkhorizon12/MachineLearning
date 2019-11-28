from tensorflow.keras.utils import get_file

fname = 'momjjang_data.csv'
origin = 'https://skettee.github.io/post/linear_regression/momjjangban_data.csv'
path = get_file(fname, origin)

import numpy as np
import pandas as pd

df = pd.read_csv(path, index_col=0)

# Check if exist NaN data
df.head()

# Data Cleansing
df.isnull().any()

## Remove row with NaN
df = df.dropna(axis=0).reset_index(drop=True)

print(df.head())

height_data = df.height
weight_data = df.weight

# Data Analyze

import matplotlib.pyplot as plt

plt.scatter(height_data, weight_data)
plt.xlabel("height (cm)")
plt.ylabel("weight (kg)")
plt.show()

# Data Transformation
x = np.array(height_data).reshape(len(height_data), 1)
y = np.array(weight_data).reshape(len(weight_data), 1)

# print('x = ', x[:10])
# print('x.shape = ', x.shape)
# print('y = ', y[:10])
# print('y.shape = ', y.shape)

# Loss Function
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d

# W, b의 범위 지정
w = np.arange(-5.0, 5.0, 0.1)
b = np.arange(-5.0, 5.0, 0.1)
j_array = []

W, B = np.meshgrid(w, b)

# w, b를 하나씩 대응한다
for we, be in zip(np.ravel(W), np.ravel(B)):
    y_hat = np.add(np.multiply(we, x), be)
    # Loss function
    mse = mean_squared_error(y_hat, y) / 2.0
    j_array.append(mse)

# 손실(Loss)를 구한다
J = np.array(j_array).reshape(W.shape)

# 서피스 그래프를 그린다
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(W, B, J, color='b', alpha=0.5)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('z')
ax.set_zticks([])
plt.show()

# Gradient Descent
def derivative(x):
    dydx = 2 * x - 2
    return dydx

epoch = 1000  # 반복 회수
learning_rate = 0.01  # alpha

xx = 3  # 초기 x값

for i in range(epoch):
    xx = xx - learning_rate * derivative(xx)

print('x for minimum y is: ', xx)