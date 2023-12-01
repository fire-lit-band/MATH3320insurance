import pandas as pd
df=pd.read_csv("insurance.csv")
from sklearn.model_selection import train_test_split
# smoker
df=df[df['smoker']=='yes']
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('train_smoker_dataset.csv', index=False)
test_df.to_csv('test_smoker_dataset.csv', index=False)
# non-smoker
df=pd.read_csv("insurance.csv")
df=df[df['smoker']=='no']
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('train_nosmoker_dataset.csv', index=False)
test_df.to_csv('test_nosmoker_dataset.csv', index=False)

