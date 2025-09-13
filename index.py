# --- Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Make plots look nicer
sns.set(style="whitegrid")

# --- Load and Explore the Dataset ---

try:
    
    # Use Iris dataset from sklearn (built-in dataset)
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame  # contains the dataset in a Pandas DataFrame

    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("⚠️ Error: File not found. Please check your dataset path.")

# Display first few rows
print("\n--- First 5 Rows ---")
print(df.head())

# Dataset info
print("\n--- Dataset Info ---")
print(df.info())

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Clean missing values if any 
df = df.dropna()

# --- Basic Data Analysis ---

# Basic statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# Group by categorical column (target/species) and calculate mean
grouped = df.groupby("target").mean()
print("\n--- Mean values by species ---")
print(grouped)

# Observation example
print("\nObservation: Different species show different average petal lengths and widths.")

# --- Data Visualization ---

# 1. Line Chart (just for demonstration, using index vs petal length)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["petal length (cm)"], label="Petal Length", color="blue")
plt.title("Line Chart: Petal Length over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart (average petal length by species)
plt.figure(figsize=(7,5))
sns.barplot(x="target", y="petal length (cm)", data=df, ci=None, palette="viridis")
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(7,5))
plt.hist(df["sepal width (cm)"], bins=15, color="orange", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot (sepal length vs petal length)
plt.figure(figsize=(7,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
