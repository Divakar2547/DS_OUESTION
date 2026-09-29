import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("bank_dataset.csv")

print("✅ Dataset Loaded")

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.dropna()
df = df.drop_duplicates()

print(df.info())

# -----------------------------
# EDA
# -----------------------------
print("\nSummary:\n", df.describe())

# -----------------------------
# DIGITAL USAGE ANALYSIS
# -----------------------------
df['Digital_Usage'] = df['UPI_Count'] + df['Card_Count']

# -----------------------------
# VISUALIZATIONS
# -----------------------------

# 1. Digital Usage by Age Group
df.groupby('Age_Group')['Digital_Usage'].mean().plot(kind='bar')
plt.title("Digital Usage by Age Group")
plt.show()

# 2. ATM Usage by Age Group
df.groupby('Age_Group')['ATM_Count'].mean().plot(kind='bar')
plt.title("ATM Usage by Age Group")
plt.show()

# 3. UPI vs ATM
plt.scatter(df['UPI_Count'], df['ATM_Count'])
plt.xlabel("UPI Usage")
plt.ylabel("ATM Usage")
plt.title("UPI vs ATM")
plt.show()

# 4. Deposits vs Withdrawals
plt.scatter(df['Deposits'], df['Withdrawals'])
plt.title("Deposits vs Withdrawals")
plt.show()

# -----------------------------
# UNUSUAL TRANSACTIONS
# -----------------------------
high_transactions = df[df['Deposits'] > 20000]
print("\n⚠ High Transaction Customers:\n", high_transactions)

# -----------------------------
# MACHINE LEARNING (KNN)
# -----------------------------

le = LabelEncoder()

df['Age_Group'] = le.fit_transform(df['Age_Group'])
df['Occupation'] = le.fit_transform(df['Occupation'])
df['City'] = le.fit_transform(df['City'])

X = df[['Age_Group','Occupation','City','UPI_Count','Card_Count','ATM_Count','Deposits','Withdrawals']]
y = (df['Digital_Usage'] > 30).astype(int)   # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))