import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# age-bmi和charges的关系
# 加载数据并筛选吸烟者
df = pd.read_csv("insurance.csv")
smokers_df = df[df['smoker'] == 'yes']

# 用于判断的临界值
threshold = 31745.265

# 生成标记（1 或 -1）
smokers_df['label'] = np.where(smokers_df['charges'] > threshold, 1, -1)

# 准备散点图数据
ages = smokers_df['age']
bmis = smokers_df['bmi']
colors = np.where(smokers_df['label'] == 1, 'red', 'green')

# 绘制散点图
plt.scatter(ages, bmis, color=colors)
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Age vs BMI of Smokers')
plt.show()