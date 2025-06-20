# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load CSV file
# df = pd.read_csv("comments.csv")
# print("Original shape:", df.shape)
# print(df.head())

# # Clean column names
# df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
# print("\nCleaned column names:", df.columns.tolist())

# # Example columns after renaming: ['id', 'comment', 'user__id', 'posted_date', 'emoji_used', 'hashtags_used_count']

# # Convert to numeric where applicable
# df['hashtags_used_count'] = pd.to_numeric(df['hashtags_used_count'], errors='coerce')
# df.dropna(subset=['hashtags_used_count'], inplace=True)

# # ------------------- ANALYSIS -------------------

# # Count emoji usage
# plt.figure(figsize=(6,4))
# sns.countplot(x='emoji_used', data=df, palette='Set2')
# plt.title("Emoji Usage in Comments")
# plt.xlabel("Used Emoji?")
# plt.ylabel("Number of Comments")
# plt.tight_layout()
# plt.show()

# # Distribution of hashtag usage
# plt.figure(figsize=(8,4))
# sns.histplot(df['hashtags_used_count'], bins=20, kde=True, color='teal')
# plt.title("Distribution of Hashtags Used")
# plt.xlabel("Hashtag Count per Comment")
# plt.tight_layout()
# plt.show()

# # Top commenters (if multiple comments per user_id)
# if 'user__id' in df.columns:
#     top_users = df['user__id'].value_counts().head(10)
#     plt.figure(figsize=(8,4))
#     sns.barplot(x=top_users.index.astype(str), y=top_users.values)
#     plt.title("Top 10 Most Active Users by Comments")
#     plt.xlabel("User ID")
#     plt.ylabel("Number of Comments")
#     plt.tight_layout()
#     plt.show()

# # Emoji usage ratio
# emoji_yes = df[df['emoji_used'].str.lower() == 'yes'].shape[0]
# emoji_no = df[df['emoji_used'].str.lower() == 'no'].shape[0]
# total = emoji_yes + emoji_no

# print(f"\nEmoji Usage:")
# print(f" - Used Emoji: {emoji_yes} ({(emoji_yes/total)*100:.2f}%)")
# print(f" - Did not use Emoji: {emoji_no} ({(emoji_no/total)*100:.2f}%)")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Load dataset
df = pd.read_csv("comments.csv")
print("Original shape:", df.shape)

# Clean column names
df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
print("Cleaned columns:", df.columns.tolist())

# Convert columns
df['hashtags_used_count'] = pd.to_numeric(df['hashtags_used_count'], errors='coerce')
df.dropna(subset=['hashtags_used_count'], inplace=True)

# Create comment length feature
df['comment_length'] = df['comment'].astype(str).apply(len)

# ------------------- 1. Emoji Usage Count -------------------
plt.figure(figsize=(6,4))
sns.countplot(x='emoji_used', data=df, palette='Set2')
plt.title("Emoji Usage in Comments")
plt.xlabel("Used Emoji?")
plt.ylabel("Number of Comments")
plt.tight_layout()
plt.show()

# ------------------- 2. Distribution of Hashtag Usage -------------------
plt.figure(figsize=(8,4))
sns.histplot(df['hashtags_used_count'], bins=20, kde=True, color='teal')
plt.title("Distribution of Hashtags Used")
plt.xlabel("Hashtag Count per Comment")
plt.tight_layout()
plt.show()

# ------------------- 3. Top 10 Most Active Users -------------------
if 'user__id' in df.columns:
    top_users = df['user__id'].value_counts().head(10)
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_users.index.astype(str), y=top_users.values, palette="crest")
    plt.title("Top 10 Most Active Users")
    plt.xlabel("User ID")
    plt.ylabel("Number of Comments")
    plt.tight_layout()
    plt.show()

# ------------------- 4. Emoji Usage Pie Chart -------------------
emoji_counts = df['emoji_used'].str.lower().value_counts()
plt.figure(figsize=(5,5))
plt.pie(emoji_counts, labels=emoji_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title("Emoji Usage Percentage")
plt.tight_layout()
plt.show()

# ------------------- 5. Comment Length Distribution -------------------
plt.figure(figsize=(8,4))
sns.histplot(df['comment_length'], bins=30, color='purple', kde=True)
plt.title("Distribution of Comment Lengths")
plt.xlabel("Comment Length")
plt.tight_layout()
plt.show()

# ------------------- 6. Comments Over Time (if posted_date usable) -------------------
if 'posted_date' in df.columns:
    df['posted_date'] = pd.to_datetime(df['posted_date'], errors='coerce')
    if df['posted_date'].notna().any():
        daily_comments = df.groupby(df['posted_date'].dt.date).size()
        plt.figure(figsize=(10,4))
        daily_comments.plot()
        plt.title("Comments Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Comments")
        plt.tight_layout()
        plt.show()

# ------------------- 7. Heatmap of Numeric Correlations -------------------
numeric_cols = df.select_dtypes(include=['number'])
plt.figure(figsize=(8,5))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------- 8. Word Frequency WordCloud -------------------
text = " ".join(df['comment'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in Comments")
plt.tight_layout()
plt.show()
