# ==========================================
# Recommendation System with LightFM
# File: eda.py
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Ratings Dataset
# -----------------------------
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# -----------------------------
# Load Movies Dataset
# -----------------------------
movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# -----------------------------
# Merge Datasets
# -----------------------------
movie_data = pd.merge(ratings, movies, on="movie_id")

# -----------------------------
# Dataset Summary
# -----------------------------
print("=" * 60)
print("Dataset Summary")
print("=" * 60)

print(f"Total Ratings : {len(movie_data)}")
print(f"Total Users   : {movie_data['user_id'].nunique()}")
print(f"Total Movies  : {movie_data['movie_id'].nunique()}")
print(f"Average Rating: {round(movie_data['rating'].mean(), 2)}")

# ==========================================================
# 1. Rating Distribution
# ==========================================================

plt.figure(figsize=(8,5))

movie_data["rating"].hist(bins=5)

plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")

plt.savefig("assets/screenshots/rating_distribution.png")
plt.show()

# ==========================================================
# 2. Top 10 Most Rated Movies
# ==========================================================

top_movies = (
    movie_data
    .groupby("title")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Most Rated Movies\n")
print(top_movies)

plt.figure(figsize=(12,6))

top_movies.sort_values().plot(kind="barh")

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Number of Ratings")
plt.ylabel("Movie")

plt.tight_layout()
plt.savefig("assets/screenshots/top_movies.png")
plt.show()

# ==========================================================
# 3. Top 10 Most Active Users
# ==========================================================

top_users = (
    movie_data
    .groupby("user_id")
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Most Active Users\n")
print(top_users)

plt.figure(figsize=(10,5))

top_users.plot(kind="bar")

plt.title("Top 10 Most Active Users")
plt.xlabel("User ID")
plt.ylabel("Ratings Given")

plt.tight_layout()
plt.savefig("assets/screenshots/top_users.png")
plt.show()

# ==========================================================
# 4. Highest Rated Movies
# ==========================================================

highest_rated = (
    movie_data
    .groupby("title")["rating"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Highest Rated Movies\n")
print(highest_rated)
