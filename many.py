import pandas as pd

df = pd.read_csv("train_nosmoker_dataset_labeled.csv")
df=df[df['cluster']==1]
# plot 3d plot of df
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['age'], df['bmi'], df['charges'], color='blue')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
plt.title('3D Scatter Plot of Original Data')
plt.show()
# using pandasgui to see the datafram
# doing standardization to df for age,bmi and charges
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# df[['age','bmi','charges']] = scaler.fit_transform(df[['age','bmi','charges']])
# using xgboost see which one is the most important fea ture
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
X_train = df[['age', 'bmi', 'children']]
Y_train = df['charges']
xgb = XGBRegressor()
# to see feature importance
xgb.fit(X_train, Y_train)
print(xgb.feature_importances_)
# to see the score

# to see the score
from sklearn.metrics import r2_score
print(r2_score(Y_train, xgb.predict(X_train)))
# test in the test set
df_test = pd.read_csv("test_nosmoker_dataset_labeled.csv")
df_test=df_test[df_test['cluster']==1]



from pandasgui import show
show(df)

