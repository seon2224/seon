# ==========================================
# MILESTONE 3 - MODELING & ADVANCED ANALYSIS
# Netflix Content Strategy Analyzer
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("netflix_feature_engineered.csv")

print(df.head())
print(df.info())

# ==========================================
# Data Preparation
# ==========================================

# Convert duration to numeric
df["duration_num"] = df["duration"].astype(str).str.extract('(\d+)')
df["duration_num"] = pd.to_numeric(df["duration_num"], errors="coerce")
df["duration_num"].fillna(df["duration_num"].median(), inplace=True)

# Select important columns
features = [
    "release_year",
    "duration_num",
    "rating",
    "country",
    "listed_in"
]

data = df[features].copy()

# Fill missing values
data["country"].fillna("Unknown", inplace=True)
data["rating"].fillna("Unknown", inplace=True)
data["listed_in"].fillna("Unknown", inplace=True)

# One Hot Encoding
data = pd.get_dummies(data)

# Replace numeric NaN with median
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# Scale Features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("Data Prepared Successfully!")

# ==========================================
# K-Means Clustering
# ==========================================

wcss = []

for i in range(1,11):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    model.fit(scaled_data)
    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("elbow_method.png")
plt.show()

# Best K = 4
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

df["Cluster"]  = clusters

# Save cluster labels
df["Cluster"] = clusters

# ==========================================
# Cluster Summary
# ==========================================

print("\n===== Cluster Summary =====")

for cluster in sorted(df["Cluster"].unique()):
    print(f"\nCluster {cluster}")
    cluster_data = df[df["Cluster"] == cluster]

    print("Content Type:")
    print(cluster_data["type"].value_counts())

    print("\nTop Countries:")
    print(cluster_data["country"].value_counts().head(5))

    print("\nTop Genres:")
    print(cluster_data["listed_in"].value_counts().head(5))

    print("-" * 50)

# ==========================================
# PCA Visualization
# ==========================================

pca = PCA(n_components=2)

components = pca.fit_transform(scaled_data)

plt.figure(figsize=(8,6))

plt.scatter(
    components[:,0],
    components[:,1],
    c=clusters,
    cmap="viridis"
)

plt.title("Netflix Content Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.savefig("clusters_pca.png")
plt.show()

# ==========================================
# Classification
# Predict Movie vs TV Show
# ==========================================

target = df["type"]

X = data

y = target

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("\nAccuracy")
print(accuracy_score(y_test,pred))

print("\nClassification Report")
print(classification_report(y_test,pred))

print("\nConfusion Matrix")
cm = confusion_matrix(y_test,pred)

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

# ==========================================
# Feature Importance
# ==========================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,8))

importance.head(15).plot(kind="barh")

plt.title("Top 15 Important Features")

plt.savefig("feature_importance.png")

plt.show()

print("\nTop Important Features")
print(importance.head(15))

# ==========================================
# Save Final Dataset
# ==========================================

df.to_csv(
    "netflix_milestone3_output.csv",
    index=False
)

print("\nMilestone 3 Completed Successfully!")