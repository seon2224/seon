import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('netflix_titles_cleaned.csv')

# --- PART 1: EXPLORATORY DATA ANALYSIS (EDA) ---

# Step 1: Content Growth Over Time
growth = df.groupby('release_year')['show_id'].count()
plt.figure(figsize=(10, 6))
growth.plot(kind='line', marker='o')
plt.title('Netflix Content Growth Over Time')
plt.xlabel('Release Year')
plt.ylabel('Number of Titles')
plt.grid(True)
plt.savefig('content_growth.png')

# Step 2: Movies vs TV Shows
type_counts = df['type'].value_counts()
plt.figure(figsize=(8, 6))
type_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Movies vs TV Shows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.savefig('content_type.png')

# Step 3: Top 10 Genres
# Split listed_in by ', ' and flatten the list
genres = df['listed_in'].str.split(', ').explode()
top_genres = genres.value_counts().head(10)
plt.figure(figsize=(10, 6))
top_genres.plot(kind='bar', color='green')
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.savefig('top_genres.png')

# Step 4: Rating Distribution
rating_counts = df['rating'].value_counts()
plt.figure(figsize=(10, 6))
rating_counts.plot(kind='bar', color='purple')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating_distribution.png')

# Step 5: Top 10 Countries
# Filter out 'Unknown' and split by ', '
countries = df[df['country'] != 'Unknown']['country'].str.split(', ').explode()
top_countries = countries.value_counts().head(10)
plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='orange')
plt.title('Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.savefig('top_countries.png')

# Step 6: Duration Analysis (Movies Only)
movies_df = df[df['type'] == 'Movie'].copy()
# Extract numeric duration (remove ' min')
movies_df['duration_num'] = movies_df['duration'].str.extract('(\d+)').astype(float)

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(movies_df['duration_num'], kde=True, bins=30)
plt.title('Movie Duration Distribution')
plt.xlabel('Duration (min)')
plt.savefig('movie_duration_hist.png')

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=movies_df['duration_num'])
plt.title('Movie Duration Boxplot')
plt.xlabel('Duration (min)')
plt.savefig('movie_duration_box.png')

# --- PART 2: FEATURE ENGINEERING ---

# Step 7: Create Content Length Category
def categorize_duration(duration):
    if duration < 60:
        return 'Short'
    elif 60 <= duration <= 120:
        return 'Medium'
    else:
        return 'Long'

# Apply the function
movies_df['duration_category'] = movies_df['duration_num'].apply(categorize_duration)

# Save the final dataset with new feature
movies_df.to_csv('netflix_movies_with_features.csv', index=False)
