import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("StudentsPerformance.csv")

# Graph 1
plt.figure(figsize=(8,5))
plt.scatter(df["reading score"], df["math score"])
plt.xlabel("Reading Score")
plt.ylabel("Math Score")
plt.title("Reading Score vs Math Score")
plt.show()

# Graph 2
plt.figure(figsize=(8,5))
plt.hist(df["math score"], bins=15)
plt.xlabel("Math Score")
plt.ylabel("Students")
plt.title("Distribution of Math Scores")
plt.show()

# Graph 3
plt.figure(figsize=(8,5))
plt.scatter(df["writing score"], df["math score"])
plt.xlabel("Writing Score")
plt.ylabel("Math Score")
plt.title("Writing Score vs Math Score")
plt.show()