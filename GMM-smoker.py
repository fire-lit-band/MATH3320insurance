import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

# Load the new dataset
df = pd.read_csv("train_smoker_dataset.csv")

# Assuming the dataset includes a 'smoker' column
# Filter for smokers

# Assuming the dataset includes 'age', 'charges', and 'bmi' columns
# Create a new array with BMI and charges (and optionally age if relevant)
# Here, I am assuming that the structure is similar to the previous dataset
data = df[['age','bmi', 'charges']].values

# Fit Gaussian Mixture Model
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(data)

# Predict clusters
predicted_clusters = gmm.predict(data)
df['cluster'] = predicted_clusters
import os
if not os.path.exists('train_smoker_dataset_labeled.csv'):
    df.to_csv('train_smoker_dataset_labeled.csv', index=False)
df=pd.read_csv('train_smoker_dataset_labeled.csv')
predicted_clusters=df['cluster'].values


# # Visualization
# for i, data_point in enumerate(data):
#     color = 'green' if predicted_clusters[i] == 1 else 'red'
#     plt.scatter(data_point[0], data_point[1], color=color)
#
# # Plot settings
# plt.xlabel('BMI')
# plt.ylabel('Charges')
# plt.title('GMM Clustering of Smokers’ Insurance Data in New Dataset')
# plt.legend(['Cluster 1', 'Cluster 2'])
# plt.show()

#----------------------------------------------------------------------

import pandas as pd
from sklearn.svm import SVC,SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Load training data
# ... [your existing training data preparation code] ...

# Train the model with grid search on the training data
known_data=df[['age','bmi']].values
clf = SVC(kernel="linear").fit(known_data, predicted_clusters )
# Best model after grid search
best_model = clf
# test SVC to check whether it can fit the data





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
charge_class_1=charges[df['cluster'] == 1]
x1= np.column_stack((age_class_1, bmi_class_1))
model1 = LinearRegression()
model1.fit(x1, charge_class_1)
predict1=model1.predict(x1)
from sklearn.metrics import r2_score
print("class1")
print(r2_score(charge_class_1, predict1))#这里ok的
plt.scatter(age_class_1, bmi_class_1, color='green', label='Class 1')


age_class_0 = age[df['cluster'] == 0]
bmi_class_0 = bmi[df['cluster'] == 0]
x0= np.column_stack((age_class_0, bmi_class_0))
charge_class_0=charges[df['cluster'] == 0]
model0 = LinearRegression()
model0.fit(x0, charge_class_0)
predict0=model0.predict(x0)
from sklearn.metrics import r2_score
print("class0")
print(r2_score(charge_class_0, predict0))
# Scatter plot



plt.scatter(age_class_0, bmi_class_0, color='red', label='Class 0')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Scatter Plot of Predicted Clusters based on Age and BMI')
plt.legend()

plt.show()
from sklearn.metrics import mean_squared_error, r2_score
# residuals = predict0 -  charge_class_0
# residual_threshold = np.abs(residuals) / np.sqrt(mean_squared_error(charge_class_0, predict0))
# filtered_indices = residual_threshold < 0.8
# print(filtered_indices.sum()/len(filtered_indices))
# age_filter=age_class_0[filtered_indices]
# bmi_filter=bmi_class_0[filtered_indices]
# charge_filter=charge_class_0[filtered_indices]
#
# X_filtered= np.column_stack((age_filter, bmi_filter))
# Y_filtered = charge_class_0[filtered_indices]
#
# # Re-train the model with filtered data
# model.fit(X_filtered, Y_filtered)
# new_predictions = model.predict(X_filtered)
#
# # Plotting filtered scatter plot
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.scatter(age_filter, bmi_filter, Y_filtered)
# plt.show()

# Report filtered R2 score
# print(f"Filtered R2 Score: {r2_score(Y_filtered, new_predictions):.2f}")
#----------------------------------------------------------------------
import plotly.graph_objects as go

# import pandas as pd
# from pandasgui import show
# show(df)

#----------------------------------------------------------------------
# test set
test=pd.read_csv('test_smoker_dataset.csv')
test_data = test[['age','bmi']].values
label=clf.predict(test_data)
import numpy as np

# Assuming label is a NumPy array with the same length as test_data
label0_indices = label == 0
label1_indices = label == 1

# Separate test_data based on labels
label0_data = test_data[label0_indices]
label1_data = test_data[label1_indices]

# Make predictions for each group
predict_test_0 = model0.predict(label0_data)
predict_test_1 = model1.predict(label1_data)

# Initialize an array to hold the combined predictions
combined_predictions = np.zeros_like(label, dtype=float)

# Assign predictions back to the corresponding indices
combined_predictions[label0_indices] = predict_test_0
combined_predictions[label1_indices] = predict_test_1

# combined_predictions now contains the predictions in the original order of test_data


# Calculate and print mean squared error
mse_value = r2_score(test['charges'], combined_predictions)
print(f"Mean Squared Error: {mse_value:.2f}")
from pandasgui import show
show(df)
