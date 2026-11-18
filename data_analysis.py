import pandas as pd

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nTOTAL CHARGES DATA TYPE:")
print(df["TotalCharges"].dtype)

print("\nMISSING VALUES AFTER CONVERSION:")
print(df.isnull().sum())
# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())

print("\nCLEANED DATASET SHAPE:")
print(df.shape)

print("\nFIRST 5 CUSTOMERS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDATASET INFORMATION:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nCHURN DISTRIBUTION:")
print(df["Churn"].value_counts())

print("\nDataset loaded successfully!")
import matplotlib.pyplot as plt
import seaborn as sns

# Churn Distribution Visualization
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn",
    palette="Set2"
)

plt.title("Contract Type vs Customer Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()