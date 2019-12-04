# 로지스틱 회귀(Logistic Regression)

from tensorflow.keras.utils import get_file

fname = 'chinchu_data.csv'
origin = 'https://skettee.github.io/post/logistic_regression/chinchu_data.csv'
path = get_file(fname, origin)

import numpy as np
import pandas as pd

df = pd.read_csv(path, index_col=0)

# 데이터를 앞에서부터 몇 개만 샘플로 보여준다
print(df.head())
# Check if exist NaN
print(df.isnull().any())

# Remove row with NaN
df = df.dropna(axis=0).sort_values('height').reset_index(drop=True)
print(df.head())

height_data = df.height
chinchu_data = df.chinchu

# 데이터 분석(Data Analysis)
# 데이터 모양 확인
import matplotlib.pyplot as plt

plt.scatter(height_data, chinchu_data)
plt.xlabel('height (cm)')
plt.ylabel('chinchu (yes o no)')
plt.show()