# 선형회귀(Linear Regression)
from tensorflow.keras.utils import get_file

fname = 'momjjang_data.csv'
origin = 'https://skettee.github.io/post/linear_regression/momjjangban_data.csv'
path = get_file(fname, origin)

import numpy as np
import pandas as pd

df = pd.read_csv(path, index_col=0)
print(df.head())

# Data Cleansing
# Check if exist NaN data
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
# numpy.ravel: flattern 처럼 다차원의 데이터를 1차원 배열로 변환
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
# 미분 함수
def derivative(x):
    dydx = 2 * x - 2
    return dydx


# 반복 회수
epoch = 1000
learning_rate = 0.01

# 초기 x값
xx = 3

for i in range(epoch):
    xx = xx - learning_rate * derivative(xx)

print('x for minimum y is: ', xx)

# 정규화
# (현재값 - 최소값) / (최대값 - 최소값)

from sklearn import preprocessing

mm_scaler = preprocessing.MinMaxScaler()
X_train = mm_scaler.fit_transform(x)
Y_train = mm_scaler.transform(y)

plt.scatter(X_train, Y_train)
plt.xlabel('scaled-height')
plt.ylabel('scaled-weight')
plt.show()

# 모델링(Modeling)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 모델을 준비한다
model = Sequential()

# 입력 변수의 개수가 1이고 출력 개수가 y = wx + b 를 생성한다
model.add(Dense(1, input_dim=1))

# Loss function과 Optimizer 선택
model.compile(loss='mean_squared_error', optimizer='sgd')

# epochs만큼 반복해서 손실값이 최저가 되도록 모델을 훈련한다
hist = model.fit(X_train, Y_train, epochs=3000, verbose=0)

plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 가중치와 편향(W와 b)의 값을 확인
w, b = model.get_weights()
w = w[0][0]
b = b[0]
print('w: ', w)
print('b: ', b)

x_scale = mm_scaler.transform(x)
y_scale = mm_scaler.transform(y)
plt.scatter(x_scale, y_scale)
plt.plot(x_scale, w * x_scale + b, 'r')
plt.xlabel('scaled-height')
plt.ylabel('scaled-weight')
plt.show()

# 해결(Solution)
input_height = 169.0

input_x = mm_scaler.transform(np.array([input_height]).reshape(-1, 1))
predict = model.predict(input_x)
predict = mm_scaler.inverse_transform(predict)

print('몸짱반에 들어갈 수 있는 몸무게는 {:.2f} kg입니다.'.format(predict[0][0]))
