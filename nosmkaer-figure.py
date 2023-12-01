import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Read data and filter
try:
    df = pd.read_csv("insurance.csv")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

df = df[df['smoker'] == 'no']

# Define features and target
X = df[['age', 'bmi', 'children']]
Y = df['charges']

# Create and train the model
model = LinearRegression()
model.fit(X, Y)

# Predict and calculate residuals
predictions = model.predict(X)
residuals = predictions - Y

# Plotting initial scatter plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(X['age'], X['bmi'], Y)
plt.show()

# Report initial R2 score
print(f"Initial R2 Score: {r2_score(Y, predictions):.2f}")

# Filter out outliers based on residuals
residual_threshold = residuals.abs() / np.sqrt(mean_squared_error(Y, predictions))
filtered_indices = residual_threshold < 0.5
X_filtered, Y_filtered = X[filtered_indices], Y[filtered_indices]

# Re-train the model with filtered data
model.fit(X_filtered, Y_filtered)
new_predictions = model.predict(X_filtered)

# Plotting filtered scatter plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(X_filtered['age'], X_filtered['bmi'], Y_filtered)
plt.show()

# Report filtered R2 score
print(f"Filtered R2 Score: {r2_score(Y_filtered, new_predictions):.2f}")
