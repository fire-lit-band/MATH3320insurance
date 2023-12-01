import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy import stats
def process_data(df):
    sex_mapping = {'female': 1, 'male': 2}
    smoker_mapping = {'yes': 1, 'no': 2}
    region_mapping = {'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4}
    df['sex'] = df['sex'].replace(sex_mapping)
    df['smoker'] = df['smoker'].replace(smoker_mapping)
    df['region'] = df['region'].replace(region_mapping)
    return df
# Load the new dataset
df = pd.read_csv("train_nosmoker_dataset.csv")

data = df[['age','bmi', 'charges']].values

# Fit Gaussian Mixture Model
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(data)

# Predict clusters
predicted_clusters = gmm.predict(data)
df['cluster'] = predicted_clusters
import os
change=False
if not os.path.exists('train_nosmoker_dataset_labeled.csv') or change:
    df.to_csv('train_nosmoker_dataset_labeled.csv', index=False)
df=pd.read_csv('train_nosmoker_dataset_labeled.csv')
predicted_clusters=df['cluster'].values


# Visualization
for i, data_point in enumerate(data):
    color = 'green' if predicted_clusters[i] == 1 else 'red'
    plt.scatter(data_point[0], data_point[1], color=color)

# Plot settings
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('GMM Clustering of Smokers’ Insurance Data in New Dataset')
plt.legend(['Cluster 1', 'Cluster 2'])
plt.show()

#----------------------------------------------------------------------

import pandas as pd
from sklearn.svm import SVC,SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Load training data
# ... [your existing training data preparation code] ...

# Train the model with grid search on the training data
known_data=df[['age','bmi']].values
# using decision tree
# from sklearn.linear_model import LogisticRegression
# clf = SVC(kernel="linear").fit(known_data, predicted_clusters )
# Best model after grid search
train_data=df[['age','bmi']].values
train_label=df['cluster'].values
from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0).fit(train_data, train_label)
# clf = SVC(kernel="linear").fit(train_data, train_label )
# using decision tree

from sklearn import tree
clf = tree.DecisionTreeClassifier()
train_df=process_data(df).drop(['cluster','charges'],axis=1)
clf = clf.fit(train_df.values, train_label)
# check accuarcy
print(accuracy_score(train_label, clf.predict(train_df)))
print(clf)



#----------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your DataFrame and it contains 'age' and 'bmi' columns
# Also assuming 'predicted_clusters' contains the classification results
from sklearn.linear_model import LinearRegression
# Extract 'age' and 'bmi' from the DataFrame
age = df['age'].values
bmi = df['bmi'].values
charges=df['charges'].values

# Separate 'age' and 'bmi' based on 'predicted_clusters'
age_class_1 = age[df['cluster'] == 1]
bmi_class_1 = bmi[df['cluster'] == 1]
chileren_class_1=charges[df['cluster'] == 1]
charge_class_1=charges[df['cluster'] == 1]
x1= np.column_stack((age_class_1, bmi_class_1, chileren_class_1))

# using xgboost to do predict(x1 here is train data)
# use xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
X_train = x1
Y_train = charge_class_1
xgb = XGBRegressor()
# doing predict and see r2_score
xgb.fit(X_train, Y_train)
predict1=xgb.predict(X_train)
from sklearn.metrics import r2_score
print("class1")
print(r2_score(Y_train, predict1))
print(len(age_class_1))#这里ok的






age_class_0 = age[df['cluster'] == 0]
bmi_class_0 = bmi[df['cluster'] == 0]
x0= np.column_stack((age_class_0, bmi_class_0))
charge_class_0=charges[df['cluster'] == 0]
model0 =LinearRegression()
model0.fit(x0, charge_class_0)
predict0=model0.predict(x0)
from sklearn.metrics import r2_score
print("class0")
print(r2_score(charge_class_0, predict0))
print(len(age_class_0))#这里ok的
# Scatter plot



plt.scatter(age_class_0, bmi_class_0, color='red', label='Class 0')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Scatter Plot of Predicted Clusters based on Age and BMI')
plt.legend()

# test part
# first use SVR to predict the cluster
# then use linear regression to predict the charge
# then use xgboost to predict the charge
# then use r2_score to see the result

test_df=pd.read_csv('test_nosmoker_dataset.csv')
test_data=test_df[['age','bmi','children']].values
# SVM to predict
y_predict=clf.predict(process_data(test_df).drop(['charges'],axis=1).values)
print(y_predict.sum())
test_df['cluster']=y_predict
# using pandasgui to see the result
from pandasgui import show
show(test_df)


df1=test_df[test_df['cluster']==1]
df0=test_df[test_df['cluster']==0]
# linear regression to predict in df0
age=df0['age'].values
bmi=df0['bmi'].values
charge=test_df['charges'].values
x0=np.column_stack((age,bmi))
# using model0
predict0=model0.predict(x0)
charges0=df0['charges'].values
print(r2_score(charges0, predict0))
# using xgboost to predict in df1


age=df1['age'].values
bmi=df1['bmi'].values
children=df1['children'].values
x1=np.column_stack((age,bmi,children))
predict1=xgb.predict(x1)
charges1=df1['charges'].values
print(r2_score(charges1, predict1))
# compose two results to see r2_score
if predict1.size!=0:
    # x0 should mantain in cluster 0
    predict=np.concatenate((predict0,predict1),axis=0)
    charge=np.concatenate((charges0,charges1),axis=0)
else:
    predict=predict0
print("test")
print(r2_score(charge, predict))



