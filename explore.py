import pandas as pd

df = pd.read_csv("Video_Games.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.dtypes)