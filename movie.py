# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import TruncatedSVD

# # ==============================
# # Load Data
# # ==============================
# ratings = pd.read_csv(
#     "ml-100k/u.data",
#     sep="\t",
#     names=["user_id", "item_id", "rating", "timestamp"]
# )

# movies = pd.read_csv(
#     "ml-100k/u.item",
#     sep="|",
#     encoding="latin-1",
#     names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL",
#            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
#            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
#            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"],
#     usecols=[0, 1]  # only item_id and title
# )

# ratings = ratings.merge(movies, on="item_id")

# # ==============================
# # User-Item Matrix
# # ==============================
# user_item_matrix = ratings.pivot_table(
#     index="user_id",
#     columns="title",
#     values="rating"
# )

# # ==============================
# # Item-Based Collaborative Filtering
# # ==============================
# item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
# item_similarity_df = pd.DataFrame(
#     item_similarity,
#     index=user_item_matrix.columns,
#     columns=user_item_matrix.columns
# )

# def recommend_item_based(user_id, top_n=5):
#     user_ratings = user_item_matrix.loc[user_id].dropna()
#     scores = {}
    
#     for movie, rating in user_ratings.items():
#         similar_movies = item_similarity_df[movie].drop(movie)
#         for sim_movie, sim_score in similar_movies.items():
#             if pd.isna(user_item_matrix.loc[user_id, sim_movie]):  # unseen movie
#                 scores[sim_movie] = scores.get(sim_movie, 0) + sim_score * rating

#     sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     return [movie for movie, score in sorted_scores[:top_n]]

# # ==============================
# # Matrix Factorization (SVD)
# # ==============================
# user_item_matrix_filled = user_item_matrix.fillna(0)

# svd = TruncatedSVD(n_components=20, random_state=42)
# matrix_reduced = svd.fit_transform(user_item_matrix_filled)

# predicted_ratings = np.dot(matrix_reduced, svd.components_)
# predicted_ratings_df = pd.DataFrame(
#     predicted_ratings,
#     index=user_item_matrix.index,
#     columns=user_item_matrix.columns
# )

# def recommend_svd(user_id, top_n=5):
#     user_row = predicted_ratings_df.loc[user_id]
#     rated_movies = user_item_matrix.loc[user_id].dropna().index
#     recommendations = user_row.drop(rated_movies).sort_values(ascending=False)
#     return recommendations.head(top_n).index.tolist()

# # ==============================
# # Evaluation: Precision@K
# # ==============================
# def precision_at_k(recommend_func, k=5, n_users=50):
#     precisions = []
#     for user_id in np.random.choice(user_item_matrix.index, size=n_users, replace=False):
#         recs = recommend_func(user_id, top_n=k)
#         actual = user_item_matrix.loc[user_id].dropna()
#         relevant = actual[actual >= 4].index  # relevant = rating >= 4
#         hits = sum([1 for r in recs if r in relevant])
#         precisions.append(hits / k)
#     return np.mean(precisions)

# # ==============================
# # Example Run
# # ==============================
# user_id_example = 1
# print("Item-based CF recommendations:", recommend_item_based(user_id_example, top_n=5))
# print("SVD recommendations:", recommend_svd(user_id_example, top_n=5))

# print("\nPrecision@5 (Item-based CF):", precision_at_k(recommend_item_based, k=5))
# print("Precision@5 (SVD):", precision_at_k(recommend_svd, k=5))


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# ==============================
# Load Data
# ==============================
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    names=["item_id", "title", "release_date", "video_release_date", "IMDb_URL",
           "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
           "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
           "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"],
    usecols=[0, 1]  # only item_id and title
)

ratings = ratings.merge(movies, on="item_id")

# ==============================
# Train/Test Split
# ==============================
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Build user-item matrices
train_matrix = train.pivot_table(index="user_id", columns="title", values="rating")
test_matrix = test.pivot_table(index="user_id", columns="title", values="rating")

# ==============================
# Item-Based Collaborative Filtering
# ==============================
item_similarity = cosine_similarity(train_matrix.T.fillna(0))
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=train_matrix.columns,
    columns=train_matrix.columns
)

def recommend_item_based(user_id, top_n=5):
    if user_id not in train_matrix.index:
        return []
    user_ratings = train_matrix.loc[user_id].dropna()
    scores = {}
    
    for movie, rating in user_ratings.items():
        similar_movies = item_similarity_df[movie].drop(movie)
        for sim_movie, sim_score in similar_movies.items():
            if pd.isna(train_matrix.loc[user_id, sim_movie]):  # unseen movie
                scores[sim_movie] = scores.get(sim_movie, 0) + sim_score * rating

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, score in sorted_scores[:top_n]]

# ==============================
# Matrix Factorization (SVD)
# ==============================
train_matrix_filled = train_matrix.fillna(0)

svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(train_matrix_filled)

predicted_ratings = np.dot(matrix_reduced, svd.components_)
predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    index=train_matrix.index,
    columns=train_matrix.columns
)

def recommend_svd(user_id, top_n=5):
    if user_id not in predicted_ratings_df.index:
        return []
    user_row = predicted_ratings_df.loc[user_id]
    rated_movies = train_matrix.loc[user_id].dropna().index
    recommendations = user_row.drop(rated_movies).sort_values(ascending=False)
    return recommendations.head(top_n).index.tolist()

# ==============================
# Evaluation: Precision@K
# ==============================
def precision_at_k(recommend_func, k=5, n_users=50):
    precisions = []
    sampled_users = np.random.choice(train_matrix.index, size=n_users, replace=False)

    for user_id in sampled_users:
        recs = recommend_func(user_id, top_n=k)
        if not recs:
            continue
        
        if user_id in test_matrix.index:
            actual = test_matrix.loc[user_id].dropna()
            relevant = actual[actual >= 4].index  # relevant = rating >= 4
            hits = sum([1 for r in recs if r in relevant])
            precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0.0

# ==============================
# Example Run
# ==============================
user_id_example = 1
print("Item-based CF recommendations:", recommend_item_based(user_id_example, top_n=5))
print("SVD recommendations:", recommend_svd(user_id_example, top_n=5))

print("\nPrecision@5 (Item-based CF):", precision_at_k(recommend_item_based, k=5))
print("Precision@5 (SVD):", precision_at_k(recommend_svd, k=5))
