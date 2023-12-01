import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example DataFrame
df=pd.read_csv("insurance.csv")
df=df[df['smoker']=='no']

# Calculate the correlation matrix
corr = df.corr()
sns.pairplot(df)

# Create a heatmap
plt.figure(figsize=(8, 6))  # Optional: specifies the size of the figure
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Show plot
plt.show()
