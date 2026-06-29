# ==========================================
# MILESTONE 2 - EDA & FEATURE ENGINEERING
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("netflix_titles_cleaned.csv")

print(df.head())
print(df.info())

# ==========================================
# STEP 1: CONTENT GROWTH OVER TIME
# ==========================================

plt.figure(figsize=(12,6))

content_growth = df['release_year'].value_counts().sort_index()

sns.lineplot(
    x=content_growth.index,
    y=content_growth.values,
    marker='o'
)

plt.title('Netflix Content Growth Over Time')
plt.xlabel('Release Year')
plt.ylabel('Number of Titles')
plt.show()


# ==========================================
# STEP 2: MOVIES VS TV SHOWS
# ==========================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x='type'
)

plt.title('Movies vs TV Shows')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.show()


# ==========================================
# STEP 3: TOP 10 GENRES
# ==========================================

genres = df['listed_in'].str.split(', ').explode()

top_genres = genres.value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_genres.values,
    y=top_genres.index
)

plt.title('Top 10 Genres on Netflix')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()


# ==========================================
# STEP 4: RATING DISTRIBUTION
# ==========================================

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    y='rating',
    order=df['rating'].value_counts().index
)

plt.title('Content Rating Distribution')
plt.xlabel('Count')
plt.ylabel('Rating')
plt.show()


# ==========================================
# STEP 5: TOP COUNTRIES
# ==========================================

countries = df['country'].dropna().str.split(', ').explode()

top_countries = countries.value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title('Top 10 Contributing Countries')
plt.xlabel('Number of Titles')
plt.ylabel('Country')
plt.show()


# ==========================================
# STEP 6: DURATION ANALYSIS
# ==========================================

# Extract numeric duration

df['duration_num'] = df['duration'].str.extract('(\d+)')
df['duration_num'] = pd.to_numeric(df['duration_num'])

# Histogram

plt.figure(figsize=(10,6))

sns.histplot(
    df['duration_num'].dropna(),
    bins=30,
    kde=True
)

plt.title('Duration Distribution')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()

# Boxplot

plt.figure(figsize=(10,2))

sns.boxplot(
    x=df['duration_num']
)

plt.title('Duration Boxplot')
plt.show()


# ==========================================
# FEATURE ENGINEERING
# ==========================================

# STEP 7: CONTENT LENGTH CATEGORY

def categorize_duration(x):
    if pd.isna(x):
        return np.nan
    elif x < 60:
        return 'Short'
    elif x <= 120:
        return 'Medium'
    else:
        return 'Long'

df['content_length_category'] = df['duration_num'].apply(categorize_duration)

print("\nContent Length Category:")
print(df['content_length_category'].value_counts())


# ==========================================
# STEP 8: RELEASE DECADE
# ==========================================

df['release_decade'] = (df['release_year'] // 10) * 10

print("\nRelease Decades:")
print(df['release_decade'].value_counts())


# ==========================================
# STEP 9: NUMBER OF GENRES
# ==========================================

df['genre_count'] = df['listed_in'].apply(
    lambda x: len(str(x).split(','))
)

print("\nGenre Count:")
print(df['genre_count'].head())


# ==========================================
# STEP 10: MOVIE / TV SHOW ENCODING
# ==========================================

df['type_encoded'] = df['type'].map({
    'Movie': 0,
    'TV Show': 1
})

print("\nEncoded Type:")
print(df[['type','type_encoded']].head())


# ==========================================
# SAVE FEATURE ENGINEERED DATASET
# ==========================================

df.to_csv(
    'netflix_feature_engineered.csv',
    index=False
)

print("\nMilestone 2 Completed Successfully!")
print("Feature engineered dataset saved.")