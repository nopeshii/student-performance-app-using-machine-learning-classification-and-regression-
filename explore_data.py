import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())