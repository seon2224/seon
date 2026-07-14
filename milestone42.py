import os
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Netflix Content Strategy Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Strategy Dashboard")
st.markdown("### Milestone 4 - Interactive Dashboard")

# ---------------------------------------------------
# LOAD DATA & MODELS (With Safety Checks)
# ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("netflix_titles_cleaned.csv")
    except FileNotFoundError:
        st.error("❌ 'netflix_titles_cleaned.csv' not found. Please ensure it is in the same directory.")
        return pd.DataFrame()

df = load_data()

# Load the trained model and encoders safely
model, country_encoder, genre_encoder = None, None, None

try:
    model = pickle.load(open("model.pkl", "rb"))
    country_encoder = pickle.load(open("country_encoder.pkl", "rb"))
    genre_encoder = pickle.load(open("genre_encoder.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"❌ Missing model or encoder file: {e.filename}. Please make sure your pickle files are in the same directory.")

# Only proceed if data loaded successfully
if not df.empty:
    # ---------------------------------------------------
    # SIDEBAR FILTERS
    # ---------------------------------------------------
    st.sidebar.header("Dashboard Filters")

    # Country Filter
    countries = sorted(df["country"].dropna().unique())
    selected_country = st.sidebar.multiselect(
        "Select Country",
        countries,
        default=[]
    )

    # Genre Filter
    genres = sorted(df["listed_in"].dropna().unique())
    selected_genre = st.sidebar.multiselect(
        "Select Genre",
        genres,
        default=[]
    )

    # Content Type Filter
    content_types = df["type"].dropna().unique()
    content_type = st.sidebar.multiselect(
        "Content Type",
        content_types,
        default=content_types
    )

    # Release Year Filter
    min_year = int(df["release_year"].min())
    max_year = int(df["release_year"].max())
    year_range = st.sidebar.slider(
        "Release Year",
        min_year,
        max_year,
        (min_year, max_year)
    )

    # ---------------------------------------------------
    # APPLY FILTERS
    # ---------------------------------------------------
    filtered_df = df.copy()

    if selected_country:
        filtered_df = filtered_df[filtered_df["country"].isin(selected_country)]

    if selected_genre:
        filtered_df = filtered_df[filtered_df["listed_in"].isin(selected_genre)]

    filtered_df = filtered_df[filtered_df["type"].isin(content_type)]
    filtered_df = filtered_df[
        (filtered_df["release_year"] >= year_range[0]) &
        (filtered_df["release_year"] <= year_range[1])
    ]

    # ---------------------------------------------------
    # DASHBOARD SUMMARY METRICS
    # ---------------------------------------------------
    total_titles = len(filtered_df)
    total_movies = len(filtered_df[filtered_df["type"] == "Movie"])
    total_tvshows = len(filtered_df[filtered_df["type"] == "TV Show"])  # Adjusted case to match standard dataset
    total_countries = filtered_df["country"].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎬 Total Titles", total_titles)
    m2.metric("🎥 Movies", total_movies)
    m3.metric("📺 TV Shows", total_tvshows)
    m4.metric("🌍 Countries", total_countries)

    # ---------------------------------------------------
    # TABS
    # ---------------------------------------------------
    tab1, tab2 = st.tabs([
        "📊 Strategic Catalog Insights",
        "🤖 AI Analytics"
    ])

    with tab1:
        st.header("📈 Netflix Content Growth Over the Years")

        yearly = (
            filtered_df.groupby("release_year")
            .size()
            .reset_index(name="Count")
        )

        fig = px.line(
            yearly,
            x="release_year",
            y="Count",
            markers=True,
            title="Netflix Content Released by Year"
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Movies vs TV Shows")
            type_count = filtered_df["type"].value_counts()
            fig_pie = px.pie(
                values=type_count.values,
                names=type_count.index,
                title="Content Type Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Top 10 Countries")
            country_count = filtered_df["country"].value_counts().head(10)
            country_df = country_count.reset_index()
            country_df.columns = ["Country", "Count"]

            fig_bar = px.bar(
                country_df,
                x="Count",
                y="Country",
                orientation="h",
                title="Top 10 Countries"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🎭 Top 10 Genres")
        genre_count = (
            filtered_df["listed_in"]
            .str.split(", ")
            .explode()
            .value_counts()
            .head(10)
        )
        genre_df = genre_count.reset_index()
        genre_df.columns = ["Genre", "Count"]

        fig_genre = px.bar(
            genre_df,
            x="Count",
            y="Genre",
            orientation="h",
            title="Top 10 Genres"
        )
        st.plotly_chart(fig_genre, use_container_width=True)

    with tab2:
        st.header("🤖 AI Analytics")

        st.subheader("Cluster Visualization")
        try:
            st.image("clusters_pca.png", use_container_width=True)
        except Exception:
            st.warning("clusters_pca.png not found")

        st.subheader("Feature Importance")
        try:
            st.image("feature_importance.png", use_container_width=True)
        except Exception:
            st.warning("feature_importance.png not found")

        st.subheader("Confusion Matrix")
        try:
            st.image("confusion_matrix.png", use_container_width=True)
        except Exception:
            st.warning("confusion_matrix.png not found")

    # ------------------------------------------------
    # Interactive Prediction
    # ------------------------------------------------
    if model and country_encoder and genre_encoder:
        st.markdown("---")
        st.header("🎯 Interactive Prediction")

        col_pred1, col_pred2, col_pred3 = st.columns(3)

        with col_pred1:
            country = st.selectbox("Country", country_encoder.classes_)
        with col_pred2:
            genre = st.selectbox("Genre", genre_encoder.classes_)
        with col_pred3:
            duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=120)

        if st.button("Predict"):
            try:
                country_value = country_encoder.transform([country])[0]
                genre_value = genre_encoder.transform([genre])[0]

                prediction = model.predict([[country_value, genre_value, duration]])[0]
                st.success(f"Predicted Content Type: **{prediction}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Interactive Prediction is unavailable because model files are missing.")