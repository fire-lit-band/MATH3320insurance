# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture



# # Scatter plots
# # Plot BMI vs Age
# plt.figure()
# plt.scatter(df_no_smokers['bmi'], df_no_smokers['age'], color='b')
# plt.show()
#
# # Plot Age vs Charges
# plt.figure()
# plt.scatter(df_no_smokers['age'], df_no_smokers['charges'], color='r')
# plt.show()
#
# # Plot BMI vs Charges
# plt.figure()
# plt.scatter(df_no_smokers['bmi'], df_no_smokers['charges'], color='r')
# plt.show()
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import mixture
#
# # Load data
# df = pd.read_csv("insurance.csv")
#
# # Filter non-smokers
# df_non_smokers = df[df['smoker'] == 'no']
#
# # Prepare data for GMM
# data_for_gmm = df_non_smokers[['age', 'charges', 'bmi']].values
#
# # Apply Gaussian Mixture Model
# gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(data_for_gmm)
# predictions = gmm.predict(data_for_gmm)
#
# # Plot results
# colors = ['red' if pred == 1 else 'green' for pred in predictions]
# plt.scatter(df_non_smokers['age'], df_non_smokers['charges'], color=colors)
# plt.show()
#
# # Print counts
# print("Count of group 1:", np.sum(predictions == 1))
# print("Count of group 0:", np.sum(predictions == 0))
import pandas as pd
import numpy as np
# use pandasgui
# import pandasgui
# pandasgui.show(df)
#Modeling
import pandas as pd
def transfer(file_path):
    df=pd.read_csv(file_path)
    sex_mapping = {'female': 0, 'male': 1}
    region_mapping = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    smoker_mapping = {'no': 0, 'yes': 1}
    df['smoker'] = df['smoker'].replace(smoker_mapping)
    df['sex'] = df['sex'].replace(sex_mapping)
    df['region'] = df['region'].replace(region_mapping)
    return df
df=transfer("train_nosmoker_dataset.csv")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import median_absolute_error





print(df)
X_train = df.drop("charges", axis=1)
y_train = df["charges"]
import lightgbm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
df2=transfer("test_nosmoker_dataset.csv")
X_test=df2.drop("charges", axis=1)
y_test=df2["charges"]
# lgb = lightgbm.LGBMRegressor(objective = 'mse')
# lgb.fit(X_train, y_train)
# lightgbm.plot_importance(lgb)
# r2_score(y_test,lgb.predict(X_test))
# # print('Error: ', r2_score(y_test, y_pred))
# plt.show()
# 选择特征和目标变量
# scaler1=preprocessing.StandardScaler().fit(X)
# X=scaler1.transform(X.values)
# scaler2=preprocessing.StandardScaler().fit(Y.values.reshape(-1,1))
# Y=scaler2.transform(Y.values.reshape(-1,1))
# data=np.hstack((X,Y))
# df=pd.DataFrame(data,columns=['age','bmi','charges'])

# 创建并训练线性回归模型


from xgboost import XGBRegressor
import optuna

def objective_xg(trial):
    """Define the objective function"""

    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.3, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        "seed" : trial.suggest_categorical('seed', [42]),
        'objective': trial.suggest_categorical('objective', ['reg:squarederror']),
    }
    model_xgb = XGBRegressor(**params)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    return mean_squared_error(y_test,y_pred)

study_xgb = optuna.create_study(direction='minimize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb.optimize(objective_xg, n_trials=50,show_progress_bar=True)
print('Best parameters', study_xgb.best_params)
xgb = XGBRegressor(**study_xgb.best_params)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print('Error: ', r2_score(y_test, y_pred))


