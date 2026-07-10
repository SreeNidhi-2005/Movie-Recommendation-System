# 🎬 CineMatch AI - Movie Recommendation System

An intelligent **Movie Recommendation System** built using **Collaborative Filtering (SVD)** on the **MovieLens 100K Dataset**. The application predicts movies a user is likely to enjoy based on historical ratings and provides personalized recommendations through an interactive **Streamlit** web application.

---

## 📌 Project Overview

This project demonstrates how recommendation systems work using **Matrix Factorization (SVD)**. It analyzes user-movie interactions and predicts ratings for unseen movies, helping users discover new content tailored to their preferences.

The project also includes:

- 📊 Exploratory Data Analysis (EDA)
- 🤖 Recommendation Model Training
- 📈 Model Evaluation
- 🌐 Interactive Streamlit Dashboard

---

## 🚀 Features

- 🎯 Personalized Movie Recommendations
- 📊 Exploratory Data Analysis (EDA)
- 📈 Rating Distribution Visualization
- 👥 User & Movie Statistics
- ⭐ Top 10 Recommended Movies
- 📉 Model Evaluation using RMSE & MAE
- 🌐 Interactive Streamlit Interface

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| Matplotlib | Data Visualization |
| Streamlit | Web Application |
| Surprise (SVD) | Recommendation Engine |
| Joblib | Model Saving & Loading |

---

## 📂 Dataset

**MovieLens 100K Dataset**

- 100,000 Movie Ratings
- 943 Users
- 1,682 Movies

Dataset Source:
https://grouplens.org/datasets/movielens/100k/

---

## 📁 Project Structure

```
Movie-Recommendation-System
│
├── app
│   └── streamlit_app.py
│
├── assets
│   └── screenshots
│
├── data
│   └── ml-100k
│
├── models
│   └── svd_model.pkl
│
├── notebooks
│
├── src
│   ├── preprocess.py
│   ├── eda.py
│   ├── train_model.py
│   ├── recommend.py
│   ├── evaluate.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Exploratory Data Analysis

The project performs several analyses on the MovieLens dataset:

- Rating Distribution
- Top Rated Movies
- Most Active Users
- Dataset Statistics

---

## 🤖 Machine Learning Model

The recommendation engine uses:

**Algorithm:** Singular Value Decomposition (SVD)

Type:

- Collaborative Filtering
- Matrix Factorization

The model predicts ratings for unseen movies and recommends the highest-rated movies to each user.

---

## 📈 Model Evaluation

Evaluation Metrics:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Example Results:

| Metric | Value |
|---------|-------|
| RMSE | ~0.93 |
| MAE | ~0.74 |

---

## 🌐 Streamlit Application

The application provides an interactive interface where users can:

- Select a User ID
- Generate Personalized Movie Recommendations
- Explore Dataset Statistics
- View EDA Visualizations

Run the application using:

```bash
streamlit run app/streamlit_app.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/SreeNidhi-2005/Movie-Recommendation-System.git
```

Navigate into the project:

```bash
cd Movie-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the recommendation model:

```bash
python src/train_model.py
```

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

---

## 📸 Screenshots

Add screenshots here after running the project.

Example:

- 🏠 Home Page
- 🎯 Recommendation Page
- 📊 Dashboard

---

## 🔮 Future Improvements

- 🎬 Movie Poster Integration
- 🎭 Genre-based Filtering
- 👤 User Login System
- ☁️ Cloud Deployment
- 🤖 Deep Learning Recommendation Models

---

## 👩‍💻 Author

**Sreenidhi Kuruba**

Integrated B.Tech in Computer Science Engineering

Mahindra University

GitHub:
https://github.com/SreeNidhi-2005

---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub!
