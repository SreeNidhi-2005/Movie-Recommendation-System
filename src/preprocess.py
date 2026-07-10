# ==========================================
# Recommendation System with LightFM
# File: preprocess.py
# ==========================================

import pandas as pd

# -----------------------------
# Load Ratings Dataset
# -----------------------------
ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# -----------------------------
# Load Movie Dataset
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
# Display Dataset Information
# -----------------------------
print("=" * 50)
print("Ratings Dataset")
print("=" * 50)

print(ratings.head())

print("\nShape:", ratings.shape)

print("\n")

print("=" * 50)
print("Movies Dataset")
print("=" * 50)

print(movies.head())

print("\nShape:", movies.shape)