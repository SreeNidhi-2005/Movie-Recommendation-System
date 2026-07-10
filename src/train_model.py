# ==========================================
# Movie Recommendation System
# Training Model using Surprise SVD
# ==========================================

import joblib

from surprise import Dataset
from surprise import Reader
from surprise import SVD

# -----------------------------------------
# Load Dataset
# -----------------------------------------

reader = Reader(line_format='user item rating timestamp',
                sep='\t',
                rating_scale=(1,5))

data = Dataset.load_from_file(
    "data/ml-100k/u.data",
    reader=reader
)

trainset = data.build_full_trainset()

print("="*60)
print("Training Recommendation Model...")
print("="*60)

# -----------------------------------------
# Train SVD Model
# -----------------------------------------

model = SVD(
    n_factors=100,
    random_state=42
)

model.fit(trainset)

print("Model Trained Successfully!")

# -----------------------------------------
# Save Model
# -----------------------------------------

joblib.dump(model,"models/svd_model.pkl")

print("Model Saved Successfully!")
print("="*60)