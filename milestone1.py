import pandas as pd
import numpy as np

def clean_and_normalize_netflix_data(input_csv):
    print("🚀 Loading original Netflix dataset...")
    df = pd.read_csv(input_csv)
    
    # 1. Fix structural column misalignment (e.g., Louis C.K. records)
    # Runtimes ('74 min', etc.) were accidentally placed in the 'rating' column, leaving 'duration' null.
    print("🛠️  Fixing structural row misalignments...")
    misclassified_idx = df[df['duration'].isnull() & df['rating'].str.contains('min', na=False)].index
    for idx in misclassified_idx:
        df.loc[idx, 'duration'] = df.loc[idx, 'rating']
        df.loc[idx, 'rating'] = 'NR'  # Set rating to Not Rated
        
    # 2. Handle missing data values (Imputation)
    print("🩹 Imputing missing categorical records...")
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['date_added'] = df['date_added'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Unknown')
    
    # 3. Strip redundant trailing/leading whitespaces from strings
    print("🧹 Cleaning whitespaces from text fields...")
    categorical_cols = ['type', 'title', 'director', 'cast', 'country', 'rating', 'duration', 'listed_in']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
        
    # Save the primary cleaned table
    cleaned_main_file = 'netflix_titles_cleaned.csv'
    df.to_csv(cleaned_main_file, index=False)
    print(f"💾 Saved main cleaned table to: '{cleaned_main_file}'")
    
    # 4. Normalize Genre Table (Decomposing multi-valued strings to 1NF)
    print("📊 Normalizing 'listed_in' to separate genres table...")
    genres_list = []
    for idx, row in df.iterrows():
        show_id = row['show_id']
        genres = row['listed_in'].split(',')
        for g in genres:
            genres_list.append({'show_id': show_id, 'genre': g.strip()})
    df_genres = pd.DataFrame(genres_list)
    df_genres.to_csv('netflix_genres_normalized.csv', index=False)
    print("💾 Saved table to: 'netflix_genres_normalized.csv'")
    
    # 5. Normalize Country Table (Splitting multi-country productions)
    print("🌍 Normalizing 'country' to separate production countries table...")
    countries_list = []
    for idx, row in df.iterrows():
        show_id = row['show_id']
        countries = row['country'].split(',')
        for c in countries:
            countries_list.append({'show_id': show_id, 'country': c.strip()})
    df_countries = pd.DataFrame(countries_list)
    df_countries.to_csv('netflix_countries_normalized.csv', index=False)
    print("💾 Saved table to: 'netflix_countries_normalized.csv'")
    
    # 6. Normalize Cast Table (Optional link table for deep performer analysis)
    print("🎭 Normalizing 'cast' to separate actors link table...")
    cast_list = []
    for idx, row in df.iterrows():
        show_id = row['show_id']
        cast_members = row['cast'].split(',')
        for member in cast_members:
            cast_list.append({'show_id': show_id, 'actor': member.strip()})
    df_cast = pd.DataFrame(cast_list)
    df_cast.to_csv('netflix_cast_normalized.csv', index=False)
    print("💾 Saved table to: 'netflix_cast_normalized.csv'")
    
    print("\n✅ Milestone 1 Data Engineering pipeline executed successfully!")

if __name__ == '__main__':
    # Make sure 'netflix_titles.csv' is present in your active workspace folder
    clean_and_normalize_netflix_data('netflix_titles.csv')