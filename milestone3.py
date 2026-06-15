# ============================================================
# NETFLIX PROJECT - MILESTONE 3 (WEEK 5 & 6)
# Modeling & Advanced Analysis
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

sns.set_style("whitegrid")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("netflix_titles_cleaned.csv")

print("="*60)
print("DATASET SHAPE")
print(df.shape)
print("="*60)

# ============================================================
# DATA PREPROCESSING
# ============================================================

df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Unknown')

# Extract Duration Number

def extract_duration(x):
    try:
        return int(str(x).split()[0])
    except:
        return np.nan

df['duration_num'] = df['duration'].apply(extract_duration)
df['duration_num'] = df['duration_num'].fillna(
    df['duration_num'].median()
)

# Extract First Genre

df['genre'] = df['listed_in'].apply(
    lambda x: str(x).split(',')[0].strip()
)

# ============================================================
# ENCODING
# ============================================================

genre_encoder = LabelEncoder()
rating_encoder = LabelEncoder()
type_encoder = LabelEncoder()

df['genre_encoded'] = genre_encoder.fit_transform(df['genre'])
df['rating_encoded'] = rating_encoder.fit_transform(df['rating'])
df['type_encoded'] = type_encoder.fit_transform(df['type'])

# ============================================================
# CLUSTERING (K-MEANS)
# ============================================================

cluster_features = df[
    ['genre_encoded',
     'duration_num',
     'rating_encoded']
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    cluster_features
)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(
    scaled_features
)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

# ============================================================
# 1. CLUSTER SCATTER PLOT
# ============================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x='duration_num',
    y='genre_encoded',
    hue='Cluster',
    palette='Set2'
)

plt.title("Netflix Content Clusters")
plt.xlabel("Duration")
plt.ylabel("Genre Encoded")

plt.tight_layout()
plt.savefig("cluster_scatter_plot.png")
plt.show()

# ============================================================
# CLASSIFICATION
# Movie vs TV Show
# ============================================================

X = df[
    ['release_year',
     'duration_num',
     'genre_encoded',
     'rating_encoded']
]

y = df['type_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n")
print("="*60)
print("CLASSIFICATION RESULTS")
print("="*60)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred
))

# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n")
print("="*60)
print("FEATURE IMPORTANCE")
print("="*60)
print(importance)

# ============================================================
# 3. FEATURE IMPORTANCE BAR CHART
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title(
    "Feature Importance for Content Type Prediction"
)

plt.tight_layout()
plt.savefig("feature_importance_chart.png")
plt.show()

# ============================================================
# 4. FEATURE IMPORTANCE TABLE
# ============================================================

plt.figure(figsize=(8,4))

plt.axis('off')

table = plt.table(
    cellText=importance.values,
    colLabels=importance.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Feature Importance Table")

plt.tight_layout()
plt.savefig("feature_importance_table.png")
plt.show()

# ============================================================
# 5. TOP COUNTRIES CHART
# ============================================================

country_count = (
    df['country']
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12,6))

country_count.plot(
    kind='bar'
)

plt.title(
    "Top 10 Countries with Netflix Content"
)

plt.xlabel("Country")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig("top_countries.png")
plt.show()

# ============================================================
# 6. TOP GENRES CHART
# ============================================================

genre_count = (
    df['genre']
    .value_counts()
    .head(10)
)

plt.figure(figsize=(12,6))

genre_count.plot(
    kind='bar'
)

plt.title("Top 10 Netflix Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig("top_genres.png")
plt.show()

# ============================================================
# 7. COUNTRY VS GENRE HEATMAP
# ============================================================

top_countries = (
    df['country']
    .value_counts()
    .head(10)
    .index
)

filtered_df = df[
    df['country'].isin(top_countries)
]

heatmap_data = pd.crosstab(
    filtered_df['country'],
    filtered_df['genre']
)

plt.figure(figsize=(15,8))

sns.heatmap(
    heatmap_data,
    cmap='YlGnBu'
)

plt.title(
    "Country vs Genre Distribution"
)

plt.tight_layout()
plt.savefig("country_genre_heatmap.png")
plt.show()

# ============================================================
# 8. CLUSTER DISTRIBUTION
# ============================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x='Cluster'
)

plt.title(
    "Distribution of Netflix Clusters"
)

plt.xlabel("Cluster")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("cluster_distribution.png")
plt.show()

# ============================================================
# CLUSTER SUMMARY
# ============================================================

cluster_summary = df.groupby(
    'Cluster'
)['duration_num'].agg(
    ['mean', 'count']
)

print("\n")
print("="*60)
print("CLUSTER SUMMARY")
print("="*60)
print(cluster_summary)

# ============================================================
# SAVE FINAL RESULTS
# ============================================================

df.to_csv(
    "netflix_milestone3_results.csv",
    index=False
)

print("\nAnalysis Completed Successfully!")
print("Results saved as: netflix_milestone3_results.csv")

print("\nGenerated Images:")
print("1. cluster_scatter_plot.png")
print("2. confusion_matrix.png")
print("3. feature_importance_chart.png")
print("4. feature_importance_table.png")
print("5. top_countries.png")
print("6. top_genres.png")
print("7. country_genre_heatmap.png")
print("8. cluster_distribution.png")

# ============================================================
# END OF PROJECT
# ============================================================