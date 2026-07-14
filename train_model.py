import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("netflix_titles_cleaned.csv")

# Keep required columns
df = df[['type', 'country', 'listed_in', 'duration']].dropna()

# Convert duration to numeric
df['duration'] = df['duration'].astype(str).str.extract('(\d+)').astype(int)

# Encode categorical columns
country_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df['country'] = country_encoder.fit_transform(df['country'])
df['listed_in'] = genre_encoder.fit_transform(df['listed_in'])

# -------------------------
# Classification Model
# -------------------------
X = df[['country', 'listed_in', 'duration']]
y = df['type']

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X, y)

joblib.dump(classifier, "classification_model.pkl")

# -------------------------
# Clustering Model
# -------------------------
clustering = KMeans(n_clusters=4, random_state=42, n_init=10)
clustering.fit(X)

joblib.dump(clustering, "clustering_model.pkl")

# Save encoders
joblib.dump(country_encoder, "country_encoder.pkl")
joblib.dump(genre_encoder, "genre_encoder.pkl")

print("All models saved successfully!")