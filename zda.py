# # 📦 Importing libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score

# # 📥 Load dataset
# df = pd.read_csv("zomato.csv", encoding="latin-1")
# print("Initial Shape:", df.shape)

# # ❌ Drop irrelevant columns
# columns_to_drop = ['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list']
# df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# # 🧹 Drop duplicates and missing values
# df.drop_duplicates(inplace=True)
# df.dropna(how='any', inplace=True)
# print("After dropping NA:", df.shape)

# # 🧼 Clean 'rate'
# df = df[df['rate'].notnull()]
# df = df[~df['rate'].isin(['NEW', '-', '\\nTop floor'])]
# df['rate'] = df['rate'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)

# # 💰 Clean 'approx_cost(for two people)'
# df['approx_cost(for two people)'] = (
#     df['approx_cost(for two people)']
#     .astype(str)
#     .str.replace(',', '', regex=False)
# )
# df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')

# # 🧼 Drop rows with remaining NaNs
# df.dropna(subset=['rate', 'approx_cost(for two people)'], inplace=True)

# # 🔢 Encode binary columns
# df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
# df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})

# # 🏷️ Label Encoding for categorical columns
# le = LabelEncoder()
# for col in ['location', 'rest_type', 'cuisines']:
#     df[col] = le.fit_transform(df[col].astype(str))

# # 🎯 Create rating category
# df['rating_category'] = pd.cut(df['rate'],
#                                bins=[0, 2.5, 3.5, 4.0, 5.0],
#                                labels=['Poor', 'Average', 'Good', 'Excellent'])

# # -------------------- 📊 Visualizations --------------------

# # 1. Online Order Availability
# plt.figure(figsize=(10,5))
# sns.countplot(data=df, x='online_order')
# plt.title("Online Order Availability")
# plt.show()

# # 2. Ratings Distribution
# plt.figure(figsize=(10,5))
# sns.histplot(df['rate'], bins=20, kde=True)
# plt.title("Distribution of Ratings")
# plt.show()

# # 3. Top 10 Locations
# plt.figure(figsize=(10,5))
# top_locations = df['location'].value_counts()[:10]
# sns.barplot(x=top_locations.index, y=top_locations.values)
# plt.title("Top 10 Locations with Most Restaurants")
# plt.xticks(rotation=45)
# plt.show()

# # 4. Correlation Heatmap (✅ FIXED)
# plt.figure(figsize=(10,6))
# sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

# # -------------------- 🤖 Machine Learning --------------------

# # 🎯 Features and target
# features = ['online_order', 'book_table', 'votes', 'location', 'rest_type', 'cuisines', 'approx_cost(for two people)']
# X = df[features]
# y = df['rating_category']

# # 🧪 Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 🌳 Train RandomForest
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # 🧾 Evaluation
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))


# 📦 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 📥 Load dataset
df = pd.read_csv("zomato.csv", encoding="latin-1")
print("Initial Shape:", df.shape)

# ❌ Drop irrelevant columns
columns_to_drop = ['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# 🧹 Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(how='any', inplace=True)
print("After dropping NA:", df.shape)

# 🧼 Clean 'rate'
df = df[df['rate'].notnull()]
df = df[~df['rate'].isin(['NEW', '-', '\\nTop floor'])]
df['rate'] = df['rate'].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)

# 💰 Clean 'approx_cost(for two people)'
df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '', regex=False)
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')
df.dropna(subset=['rate', 'approx_cost(for two people)'], inplace=True)

# 🔢 Encode binary columns
df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})

# 🏷️ Label Encoding
le = LabelEncoder()
for col in ['location', 'rest_type', 'cuisines']:
    df[col] = le.fit_transform(df[col].astype(str))

# 🎯 Rating category
df['rating_category'] = pd.cut(df['rate'],
                               bins=[0, 2.5, 3.5, 4.0, 5.0],
                               labels=['Poor', 'Average', 'Good', 'Excellent'])

# -------------------- 📊 10 Visualizations --------------------

# 1. Online Order Availability
plt.figure(figsize=(8,4))
sns.countplot(data=df, x='online_order')
plt.title("Online Order Availability")
plt.show()

# 2. Book Table vs Rating
plt.figure(figsize=(8,4))
sns.boxplot(data=df, x='book_table', y='rate')
plt.title("Book Table vs Rating")
plt.show()

# 3. Ratings Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['rate'], bins=20, kde=True)
plt.title("Distribution of Ratings")
plt.show()

# 4. Top 10 Locations
plt.figure(figsize=(10,5))
top_locations = df['location'].value_counts().head(10)
sns.barplot(x=top_locations.index, y=top_locations.values)
plt.title("Top 10 Locations with Most Restaurants")
plt.xticks(rotation=45)
plt.show()

# 5. Votes vs Rating
plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='votes', y='rate')
plt.title("Votes vs Rating")
plt.show()

# 6. Cost vs Rating
plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='approx_cost(for two people)', y='rate')
plt.title("Cost for Two vs Rating")
plt.show()

# 7. Rating Category Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='rating_category', data=df, palette='Set2')
plt.title("Rating Category Distribution")
plt.show()

# 8. Restaurant Types (Top 10)
plt.figure(figsize=(10,5))
top_rest_types = df['rest_type'].value_counts().head(10)
sns.barplot(x=top_rest_types.index, y=top_rest_types.values)
plt.title("Top 10 Restaurant Types")
plt.xticks(rotation=45)
plt.show()

# 9. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 10. Cuisine Types (Top 10)
plt.figure(figsize=(10,5))
top_cuisines = df['cuisines'].value_counts().head(10)
sns.barplot(x=top_cuisines.index, y=top_cuisines.values)
plt.title("Top 10 Cuisines")
plt.xticks(rotation=45)
plt.show()

# -------------------- 🤖 Machine Learning --------------------

# Features and target
features = ['online_order', 'book_table', 'votes', 'location', 'rest_type', 'cuisines', 'approx_cost(for two people)']
X = df[features]
y = df['rating_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
