# -*- coding: utf-8 -*-

# https://skettee.github.io/post/logistic_regression/
# 로지스틱 회귀(Logistic Regression)

from tensorflow.keras.utils import get_file

fname = 'chinchu_data.csv'
origin = 'https://skettee.github.io/post/logistic_regression/chinchu_data.csv'
path = get_file(fname, origin)

import numpy as np
import pandas as pd

df = pd.read_csv(path, index_col=0)

# 데이터를 앞에서부터 몇 개만 샘플로 보여준다
# print(df.head())
# Check if exist NaN
# print(df.isnull().any())

# Remove row with NaN
df = df.dropna(axis=0).sort_values('height').reset_index(drop=True)
# print(df.head())

height_data = df.height
chinchu_data = df.chinchu

# 데이터 분석(Data Analysis)
# 데이터 모양 확인
import matplotlib.pyplot as plt

# plt.scatter(height_data, chinchu_data)
# plt.xlabel('height (cm)')
# plt.ylabel('chinchu (yes o no)')
# plt.show()

# 데이터를 보니 선형회귀로는 답이 안나올 것 같음

# 데이터 변환(Data Transformation)
# 열: 데이터 개수, 칼럼: 측정한 항목의 개수

def transform_y(y):
    if y == 'yes':
        return 1
    else:
        return 0

import numpy as np

x = np.array(height_data).reshape(len(height_data), 1)
y = np.array([transform_y(i) for i in chinchu_data ]).reshape(len(chinchu_data), 1)

# plt.scatter(x, y)
# plt.xlabel('height (cm)')
# plt.ylabel('chinchu')
# plt.show()

# 로지스틱 모델링(Logistic Modeling)
# 1. 계단함수(Step Function)

# plt.step(x, y)
# plt.show()

from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d

# w, b의 범위를 결정
w = np.arange(-10, 10, 0.1)
d = np.arange(160, 180, 1)
j_array = []

# (20, 200) 매트릭스로 변환
W, D = np.meshgrid(w, d)

# w, b를 하나씩 대응
for we, de in zip(np.ravel(W), np.ravel(D)):
    z_hat = np.multiply(we, x)
    y_list = []
    for ze in z_hat:
        if ze < de:
            y_list.append(0)
        else:
            y_list.append(1)
    y_hat = np.array(y_list)
    # Cost Function
    mse = mean_squared_error(y_hat, y) / 2.0
    j_array.append(mse)

# 손실(Loss)을 구하고, (20, 200) 매트릭스로 변환
J = np.array(j_array).reshape(W.shape)

# 서피스 그래프를 그린다
# fig = plt.figure()
# ax = plt.axes(projection="3d")
#
# ax.plot_surface(W, D, J, color='b', alpha=0.5)
# ax.set_xlabel('w')
# ax.set_ylabel('d')
# ax.set_zlabel('J')
# plt.show()
# 계단함수는 미분이 불가능하기 때문에 사용할 수 없다
# 2. 활성화함수: 시그모이드 함수(Sigmoid Funciton)

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# linspace(start, stop, num=50): num 수만큼 균일하게 start, stop 사이를 채운다
# xx = np.linspace(-10, 10, 100)
#
# plt.plot(xx, sigmoid(xx))
# plt.show()

# 선형회귀에 쓰인 평균제곱오차가 아닌,
# 분류를 위한 손실함수: Cross_entropy Loss 사용
from sklearn.metrics import log_loss

cross_entropy_loss = True

# W, b의 범위 결정
w = np.arange(20, 30, 0.1)
b = np.arange(-4595, -4585, 0.1)

j_loss = []

# 매트릭스로 변환
W, B = np.meshgrid(w, b)

# w, b를 하나씩 대응
for we, be in zip(np.ravel(W), np.ravel(B)):
    z = np.add(np.multiply(we, x), be)
    y_hat = sigmoid(z)

    # Loss Function
    if cross_entropy_loss:
        # Log loss, aka logistic loss or cross-entropy loss.
        loss = log_loss(y, y_hat)
    else:
        # Mean squred error
        loss = mean_squared_error(y_hat, y) / 2.0

    j_loss.append(loss)

# 손실을 구한다
J = np.array(j_loss).reshape(W.shape)

# 서피스 그래프를 그린다
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot_surface(W, B, J, color='b', alpha=0.5)
# ax.set_xlabel('w')
# ax.set_ylabel('b')
# ax.set_zlabel('J')
# plt.show()

# 매개변수 경신(Optimizer): RMSProp 적용
# : 급경사인 경우에는 보폭을 낮추어서 가장 아래인지를 세밀히 살피고, 완만한 경사인 경우에는 보폭을 넓혀서 빨리 지나가는 방식

### 로지스틱 모델 정리
# 1. z = Wx + b 함수 정리
# 2. a = 활성화함수(z)
# 3. y = a
# 4. 손실 함수 정의: Cross-cross_entropy_loss
# 5. 옵티마이저 선택: RMSProp
# 6. 반복할 회수(epoch) 결정
# 7. 주어진 조건으로 모델 최적화

# 정규화(Normalization)
# 정규값 = (현재값 - 최소값) / (최대값 - 최소값)

from sklearn import preprocessing

mm_scaler = preprocessing.MinMaxScaler()
X_train = mm_scaler.fit_transform(x)
Y_train = y

# plt.scatter(X_train, Y_train)
# plt.xlabel('scaled-height')
# plt.ylabel('chinchu')
# plt.show()

# 모델링(Modeling)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation

# 모델 준비
model = Sequential()

# 입력 변수의 개수가 1이고 출력 개수가 1인 y = sigmoid(wx + b) 생성
model.add(Dense(1, input_dim=1, activation='sigmoid'))

# Loss Function과 Optimizer를 선택
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# epochs만큼 반복해서 손실값이 최저가 되도록 모델을 훈련
hist = model.fit(X_train, Y_train, epochs=10000, batch_size=20, verbose=0)

plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()