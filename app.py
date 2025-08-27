# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import TruncatedSVD

# # ==============================
# # Load Data
# # ==============================
# @st.cache_data
# def load_data():
#     ratings = pd.read_csv(
#         "ml-100k/u.data",
#         sep="\t",
#         names=["user_id", "item_id", "rating", "timestamp"]
#     )

#     movies = pd.read_csv(
#         "ml-100k/u.item",
#         sep="|",
#         encoding="latin-1",
#         names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL",
#             "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
#             "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
#             "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
#     )
#     return ratings, movies

# ratings, movies = load_data()
# ratings = ratings.merge(movies[["item_id", "title"]], on="item_id")

# # User-Item Matrix
# user_item_matrix = ratings.pivot_table(
#     index="user_id", columns="title", values="rating"
# )

# # Item similarity
# item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
# item_similarity_df = pd.DataFrame(
#     item_similarity,
#     index=user_item_matrix.columns,
#     columns=user_item_matrix.columns
# )

# # SVD
# user_item_matrix_filled = user_item_matrix.fillna(0)
# svd = TruncatedSVD(n_components=20, random_state=42)
# matrix_reduced = svd.fit_transform(user_item_matrix_filled)
# predicted_ratings = np.dot(matrix_reduced, svd.components_)
# predicted_ratings_df = pd.DataFrame(
#     predicted_ratings,
#     index=user_item_matrix.index,
#     columns=user_item_matrix.columns
# )

# # ==============================
# # Recommendation Functions
# # ==============================
# def recommend_item_based(movie_title, top_n=5, genre_filter=None):
#     if movie_title not in item_similarity_df.columns:
#         return []

#     similar_movies = item_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+20]
#     recs = similar_movies.index.tolist()

#     if genre_filter:
#         genre_movies = movies[movies[genre_filter] == 1]["title"].values
#         recs = [m for m in recs if m in genre_movies]

#     return recs[:top_n]

# def recommend_svd(user_id=1, top_n=5, genre_filter=None):
#     user_row = predicted_ratings_df.loc[user_id]
#     rated_movies = user_item_matrix.loc[user_id].dropna().index
#     recs = user_row.drop(rated_movies).sort_values(ascending=False).index.tolist()

#     if genre_filter:
#         genre_movies = movies[movies[genre_filter] == 1]["title"].values
#         recs = [m for m in recs if m in genre_movies]

#     return recs[:top_n]

# def recommend_hybrid(movie_title=None, user_id=1, top_n=5, genre_filter=None):
#     rec_item = recommend_item_based(movie_title, top_n=top_n*2, genre_filter=genre_filter) if movie_title else []
#     rec_svd = recommend_svd(user_id, top_n=top_n*2, genre_filter=genre_filter)
#     combined = pd.Series(rec_item + rec_svd).value_counts().index.tolist()
#     return combined[:top_n]

# # ==============================
# # Streamlit App
# # ==============================
# st.title("🎬 Movie Recommendation System")

# st.sidebar.header("User Input")
# movie_input = st.sidebar.text_input("Enter a movie you like (optional):")
# genre_input = st.sidebar.selectbox(
#     "Pick a genre (optional):", 
#     ["None"] + [g for g in movies.columns[5:]]
# )

# genre_filter = None if genre_input == "None" else genre_input

# method = st.sidebar.radio("Choose recommendation method:", ["Item-based CF", "SVD", "Hybrid"])
# top_n = st.sidebar.slider("Number of recommendations:", 3, 15, 5)

# # Generate Recommendations
# if method == "Item-based CF" and movie_input:
#     recommendations = recommend_item_based(movie_input, top_n, genre_filter)
# elif method == "SVD":
#     recommendations = recommend_svd(1, top_n, genre_filter)
# else:
#     recommendations = recommend_hybrid(movie_input if movie_input else None, 1, top_n, genre_filter)

# st.subheader("Your Recommendations 🍿")
# if recommendations:
#     for i, r in enumerate(recommendations, 1):
#         st.write(f"{i}. {r}")
# else:
#     st.write("No recommendations found. Try another movie or genre.")

# # ==============================
# # Visualizations
# # ==============================
# st.subheader("📊 Data Visualizations")

# # Ratings Distribution
# fig, ax = plt.subplots()
# ratings["rating"].hist(ax=ax, bins=5, edgecolor="black")
# ax.set_title("Ratings Distribution")
# ax.set_xlabel("Rating")
# ax.set_ylabel("Frequency")
# st.pyplot(fig)

# # Top Genres
# genre_counts = movies.iloc[:, 5:].sum().sort_values(ascending=False)
# fig, ax = plt.subplots()
# sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax)
# ax.set_title("Number of Movies per Genre")
# st.pyplot(fig)

# # Similarity Heatmap (small subset)
# sample_movies = movies["title"].sample(20, random_state=42)
# sim_subset = item_similarity_df.loc[sample_movies, sample_movies]
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(sim_subset, cmap="coolwarm", ax=ax)
# ax.set_title("Movie Similarity Heatmap (sample)")
# st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    movies = pd.read_csv(
        "ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL",
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    )

    ratings = pd.read_csv(
        "ml-100k/u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    return ratings, movies

ratings, movies = load_data()
ratings = ratings.merge(movies[["item_id", "title"]], on="item_id")

# User-Item Matrix
user_item_matrix = ratings.pivot_table(
    index="user_id", columns="title", values="rating"
)

# Item similarity
item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

# SVD
user_item_matrix_filled = user_item_matrix.fillna(0)
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(user_item_matrix_filled)
predicted_ratings = np.dot(matrix_reduced, svd.components_)
predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    index=user_item_matrix.index,
    columns=user_item_matrix.columns
)

# ==============================
# Recommendation Functions
# ==============================
def recommend_item_based(top_n=5, genre_filter=None):
    # Average similarity scores
    avg_scores = item_similarity_df.mean().sort_values(ascending=False)
    recs = avg_scores.index.tolist()

    if genre_filter:
        genre_movies = movies[movies[genre_filter].sum(axis=1) > 0]["title"].values
        recs = [m for m in recs if m in genre_movies]

    return recs[:top_n]

def recommend_svd(user_id=1, top_n=5, genre_filter=None):
    user_row = predicted_ratings_df.loc[user_id]
    rated_movies = user_item_matrix.loc[user_id].dropna().index
    recs = user_row.drop(rated_movies).sort_values(ascending=False).index.tolist()

    if genre_filter:
        genre_movies = movies[movies[genre_filter].sum(axis=1) > 0]["title"].values
        recs = [m for m in recs if m in genre_movies]

    return recs[:top_n]

def recommend_hybrid(user_id=1, top_n=5, genre_filter=None):
    rec_item = recommend_item_based(top_n=top_n*2, genre_filter=genre_filter)
    rec_svd = recommend_svd(user_id, top_n=top_n*2, genre_filter=genre_filter)
    combined = pd.Series(rec_item + rec_svd).value_counts().index.tolist()
    return combined[:top_n]

# ==============================
# Streamlit App
# ==============================
st.title("🎬 90s Movie Recommender")
st.markdown("This app recommends movies from the **MovieLens 100K dataset (mostly 90s, up to 1998)** "
            "using three methods: Item-based CF, SVD, and Hybrid.")

st.sidebar.header("User Input")

# Multiple genres
genre_input = st.sidebar.multiselect(
    "Pick one or more genres (optional):", 
    [g for g in movies.columns[5:]]
)
genre_filter = genre_input if genre_input else None

method = st.sidebar.radio("Choose recommendation method:", ["Item-based CF", "SVD", "Hybrid"])
top_n = st.sidebar.slider("Number of recommendations:", 3, 15, 5)

# Button to trigger
if st.sidebar.button("🔍 Get Recommendations"):
    if method == "Item-based CF":
        recommendations = recommend_item_based(top_n, genre_filter)
    elif method == "SVD":
        recommendations = recommend_svd(1, top_n, genre_filter)
    else:
        recommendations = recommend_hybrid(1, top_n, genre_filter)

    st.subheader("📌 Your Recommendations 🍿")
    if recommendations:
        for i, r in enumerate(recommendations, 1):
            st.write(f"{i}. {r}")
    else:
        st.warning("No recommendations found for the selected filters.")

# ==============================
# Visualizations
# ==============================
st.subheader("📊 Data Visualizations")

# Ratings Distribution
fig, ax = plt.subplots()
ratings["rating"].hist(ax=ax, bins=5, edgecolor="black")
ax.set_title("Ratings Distribution")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Top Genres
genre_counts = movies.iloc[:, 5:].sum().sort_values(ascending=False)
fig, ax = plt.subplots()
sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax)
ax.set_title("Number of Movies per Genre")
st.pyplot(fig)

# Similarity Heatmap (small subset)
sample_movies = movies["title"].sample(15, random_state=42)
sim_subset = item_similarity_df.loc[sample_movies, sample_movies]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sim_subset, cmap="coolwarm", ax=ax)
ax.set_title("Movie Similarity Heatmap (sample)")
st.pyplot(fig)
