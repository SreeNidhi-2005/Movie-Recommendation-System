import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="🎬 CineMatch AI",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = joblib.load("models/svd_model.pkl")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

ratings = pd.read_csv(
    "data/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "data/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0,1],
    names=["movie_id","title"]
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("🎬 CineMatch AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎯 Recommendations",
        "📊 Dataset Dashboard",
        "ℹ About"
    ]
)

# ==========================================================
# HOME
# ==========================================================

if page == "🏠 Home":

    st.title("🎬 CineMatch AI")

    st.markdown(
        """
        ### Personalized Movie Recommendation System

        This application recommends movies to users using
        **Collaborative Filtering (SVD)** trained on the
        **MovieLens 100K Dataset**.
        """
    )

    st.divider()

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "👥 Users",
        ratings["user_id"].nunique()
    )

    col2.metric(
        "🎬 Movies",
        movies["movie_id"].nunique()
    )

    col3.metric(
        "⭐ Ratings",
        len(ratings)
    )

    st.divider()

    st.subheader("Project Highlights")

    st.success("✔ Collaborative Filtering Recommendation System")

    st.success("✔ MovieLens 100K Dataset")

    st.success("✔ Surprise SVD Algorithm")

    st.success("✔ Streamlit Interactive Dashboard")

    st.success("✔ Personalized Top-10 Recommendations")

# ==========================================================
# RECOMMENDATIONS
# ==========================================================

elif page == "🎯 Recommendations":

    st.title("🎯 Movie Recommendations")

    user_id = st.selectbox(
        "Select User ID",
        sorted(ratings["user_id"].unique())
    )

    if st.button("Recommend Movies"):

        rated_movies = ratings[
            ratings["user_id"] == user_id
        ]["movie_id"].tolist()

        all_movies = movies["movie_id"].tolist()

        unrated_movies = [
            movie
            for movie in all_movies
            if movie not in rated_movies
        ]

        predictions = []

        for movie in unrated_movies:

            prediction = model.predict(user_id,movie)

            predictions.append(
                (movie,prediction.est)
            )

        predictions.sort(
            key=lambda x:x[1],
            reverse=True
        )

        st.subheader("🏆 Top 10 Recommended Movies")

        results=[]

        rank=1

        medals={
            1:"🥇",
            2:"🥈",
            3:"🥉"
        }

        for movie_id,score in predictions[:10]:

            movie_name=movies.loc[
                movies["movie_id"]==movie_id,
                "title"
            ].values[0]

            emoji=medals.get(rank,str(rank))

            results.append({
                "Rank":emoji,
                "Movie":movie_name,
                "Predicted Rating":round(score,2)
            })

            rank+=1

        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )

# ==========================================================
# DASHBOARD
# ==========================================================

elif page == "📊 Dataset Dashboard":

    st.title("📊 Dataset Dashboard")

    st.subheader("Rating Distribution")

    st.image(
        "assets/screenshots/rating_distribution.png",
        use_container_width=True
    )

    st.subheader("Top Rated Movies")

    st.image(
        "assets/screenshots/top_movies.png",
        use_container_width=True
    )

    st.subheader("Most Active Users")

    st.image(
        "assets/screenshots/top_users.png",
        use_container_width=True
    )

# ==========================================================
# ABOUT
# ==========================================================

else:

    st.title("ℹ About Project")

    st.markdown("""
## 🎬 CineMatch AI

### Algorithm

- Collaborative Filtering
- Matrix Factorization
- Surprise SVD

### Dataset

MovieLens 100K Dataset

### Tech Stack

- Python
- Pandas
- Streamlit
- Surprise
- Joblib

### Features

- Personalized Movie Recommendations
- Dataset Analytics
- Interactive Dashboard
- Top-10 Movie Predictions

---

### Developed By

**Sreenidhi Kuruba**

Integrated B.Tech CSE
Mahindra University
""")