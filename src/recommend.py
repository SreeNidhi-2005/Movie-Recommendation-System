# ==========================================
# Movie Recommendation System
# Recommend Movies for a User
# ==========================================

import joblib
import pandas as pd

# -----------------------------------------
# Load Trained Model
# -----------------------------------------

model = joblib.load("models/svd_model.pkl")

# -----------------------------------------
# Load Ratings
# -----------------------------------------

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# -----------------------------------------
# Load Movie Names
# -----------------------------------------

movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# -----------------------------------------
# User Input
# -----------------------------------------

user_id = int(input("Enter User ID (1-943): "))

# Movies already rated by the user
rated_movies = ratings[
    ratings["user_id"] == user_id
]["movie_id"].tolist()

# Movies not rated
all_movies = movies["movie_id"].tolist()

unrated_movies = [
    movie for movie in all_movies
    if movie not in rated_movies
]

# -----------------------------------------
# Predict Ratings
# -----------------------------------------

predictions = []

for movie in unrated_movies:

    prediction = model.predict(
        uid=user_id,
        iid=movie
    )

    predictions.append(
        (movie, prediction.est)
    )

# Sort by predicted rating
predictions.sort(
    key=lambda x: x[1],
    reverse=True
)

# -----------------------------------------
# Display Top 10 Recommendations
# -----------------------------------------

print("\n" + "=" * 60)
print(f"Top 10 Recommended Movies for User {user_id}")
print("=" * 60)

for movie_id, predicted_rating in predictions[:10]:

    movie_name = movies.loc[
        movies["movie_id"] == movie_id,
        "title"
    ].values[0]

    print(f"{movie_name}  ⭐ Predicted Rating: {predicted_rating:.2f}")