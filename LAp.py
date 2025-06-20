# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("loan_prediction.csv")
print("Initial Dataset Shape:", df.shape)

# ----------------- 🧹 Data Cleaning -----------------

# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Drop Loan_ID
df.drop('Loan_ID', axis=1, inplace=True)

# ----------------- 🔍 Data Visualization (10 Diagrams) -----------------

# 1. Loan Approval Status Count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Loan_Status', palette='Set2')
plt.title("Loan Approval Status")
plt.show()

# 2. Education vs Loan Approval
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Education', hue='Loan_Status', palette='Set1')
plt.title("Loan Status by Education")
plt.show()

# 3. Income vs Loan Status
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Loan_Status', y='ApplicantIncome')
plt.title("Applicant Income by Loan Status")
plt.show()

# 4. LoanAmount Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['LoanAmount'], bins=30, kde=True, color='purple')
plt.title("Loan Amount Distribution")
plt.show()

# 5. Property Area vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Property_Area', hue='Loan_Status', palette='coolwarm')
plt.title("Loan Status by Property Area")
plt.show()

# 6. Credit History Count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Credit_History', palette='pastel')
plt.title("Credit History Count")
plt.show()

# 7. Self Employed vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Self_Employed', hue='Loan_Status', palette='Set3')
plt.title("Loan Status by Self Employment")
plt.show()

# 8. Coapplicant Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['CoapplicantIncome'], bins=30, color='orange', kde=True)
plt.title("Coapplicant Income Distribution")
plt.show()

# 9. Dependents vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Dependents', hue='Loan_Status', palette='viridis')
plt.title("Loan Status by Number of Dependents")
plt.show()

# 10. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ----------------- 🔄 Label Encoding -----------------
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# Convert '3+' dependents to 3
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# ----------------- 🧠 Machine Learning -----------------

# Features & Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------- 📈 Evaluation -----------------

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
