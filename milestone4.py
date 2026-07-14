import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

st.set_page_config(page_title="Netflix Content Strategy Dashboard", layout="wide")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("netflix_titles_cleaned.csv")

df = load_data()

# ----------------------------
# Load Models
# ----------------------------
classifier = joblib.load("classification_model.pkl")
clustering = joblib.load("clustering_model.pkl")
country_encoder = joblib.load("country_encoder.pkl")
genre_encoder = joblib.load("genre_encoder.pkl")

# ----------------------------
# Prepare Data
# ----------------------------
df["duration_num"] = df["duration"].astype(str).str.extract(r"(\d+)").astype(float)

X = pd.DataFrame({
    "country": country_encoder.transform(df["country"]),
    "listed_in": genre_encoder.transform(df["listed_in"]),
    "duration": df["duration_num"]
})

df["Cluster"] = clustering.predict(X)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Netflix Filters")

country_list = sorted(
    set(
        country.strip()
        for row in df["country"].dropna().astype(str)
        for country in row.split(",")
    )
)

countries = st.sidebar.multiselect(
    "Country",
    country_list,
)

genre_list = sorted(
    set(
        genre.strip()
        for row in df["listed_in"].dropna().astype(str)
        for genre in row.split(",")
    )
)

genres = st.sidebar.multiselect(
    "Genre",
    genre_list,
)

types = st.sidebar.multiselect(
    "Content Type",
    sorted(df["type"].unique())
)

filtered_df = df.copy()

if countries:
    filtered_df = filtered_df[
        filtered_df["country"].apply(
            lambda x: any(c.strip() in countries for c in str(x).split(","))
        )
    ]

if genres:
    filtered_df = filtered_df[
        filtered_df["listed_in"].apply(
            lambda x: any(g.strip() in genres for g in str(x).split(","))
        )
    ]

if types:
    filtered_df = filtered_df[
        filtered_df["type"].isin(types)
    ]
# ----------------------------
# Title
# ----------------------------
st.set_page_config(
    page_title="Netflix Content Strategy Dashboard",
    page_icon="🎬",
    layout="wide"
)
st.title("🎬 Netflix Content Strategy Dashboard")
st.markdown("### Milestone 4 - Interactive Dashboard")

# Small text
st.caption(f"Filtered Rows: {filtered_df.shape[0]}")

# KPI Cards
total_titles = len(filtered_df)
movies = len(filtered_df[filtered_df["type"] == "Movie"])
tv_shows = len(filtered_df[filtered_df["type"] == "TV Show"])
countries = filtered_df["country"].dropna().str.split(",").explode().nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📚 Total Titles", total_titles)

with col2:
    st.metric("🎬 Movies", movies)

with col3:
    st.metric("📺 TV Shows", tv_shows)

with col4:
    st.metric("🌍 Countries", countries)

# OR a small metric box
# st.metric("Filtered Rows", filtered_df.shape[0])

tab1, tab2 = st.tabs(["Strategic Insights", "AI Analytics"])

# =====================================================
# TAB 1
# =====================================================
with tab1:

    st.subheader("Movies vs TV Shows")

    fig = px.histogram(filtered_df, x="type")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Countries")

top_country = filtered_df["country"].value_counts().head(10).reset_index()
top_country.columns = ["Country", "Titles"]

fig = px.bar(
    top_country,
    x="Country",
    y="Titles"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Genres")
top_genre = filtered_df["listed_in"].value_counts().head(10).reset_index()
top_genre.columns = ["Genre", "Titles"]

fig = px.bar(
    top_genre,
    x="Genre",
    y="Titles"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2
# =====================================================
with tab2:

    st.subheader("Cluster Distribution")

    fig = px.histogram(filtered_df, x="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Explorer")

    cluster = st.selectbox(
        "Select Cluster",
        sorted(filtered_df["Cluster"].unique())
    )

    st.dataframe(filtered_df[filtered_df["Cluster"]==cluster])

    st.subheader("Predict Content Type")

    country = st.selectbox(
        "Country",
        sorted(df["country"].unique())
    )

    genre = st.selectbox(
        "Genre",
        sorted(df["listed_in"].unique())
    )

    duration = st.slider(
        "Duration",
        30,
        300,
        120
    )

    if st.button("Predict"):

        pred = classifier.predict([[
            country_encoder.transform([country])[0],
            genre_encoder.transform([genre])[0],
            duration
        ]])[0]

        st.success(f"Predicted Content Type: {pred}")

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature":["Country","Genre","Duration"],
        "Importance":classifier.feature_importances_
    })

    fig = px.bar(
        importance,
        x="Feature",
        y="Importance"
    )

    st.plotly_chart(fig, use_container_width=True)