from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据并筛选非吸烟者
df = pd.read_csv("insurance.csv")
df = df[df['smoker'] == 'no']

# 选择特征和目标变量
X = df[['age', 'bmi', 'children']]
Y = df['charges']

# 创建并训练线性回归模型
regr = linear_model.LinearRegression()
regr.fit(X, Y)

# 绘制原始数据的3D散点图
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X['age'], X['bmi'], Y, color='blue')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
plt.title('3D Scatter Plot of Original Data')
plt.show()

# 预测并计算残差
predict = regr.predict(X)
residual = predict - Y

# 计算并打印 R² 分数
print("R² score of original model:", r2_score(Y, predict))

# 标准化残差
residual = residual / np.sqrt(mean_squared_error(Y, predict))
residual = pd.DataFrame(residual, columns=['residual'])

# 基于残差筛选数据
filtered_df = df[(residual['residual'] < 0.5) & (residual['residual'] > -0.5)]

# 重新选择筛选后的特征和目标变量
X_filtered = filtered_df[['age', 'bmi', 'children']]
Y_filtered = filtered_df['charges']

# 绘制筛选后数据的3D散点图
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_filtered['age'], X_filtered['bmi'], Y_filtered, color='red')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
plt.title('3D Scatter Plot of Filtered Data')
plt.show()

# 对筛选后的数据重新进行线性回归分析
regr.fit(X_filtered, Y_filtered)
predict_filtered = regr.predict(X_filtered)

# 计算并打印筛选后模型的 R² 分数
print("R² score of filtered model:", r2_score(Y_filtered, predict_filtered))