# ==========================================
# Evaluate Recommendation Model
# ==========================================

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

# -----------------------------------------
# Load Dataset
# -----------------------------------------

reader = Reader(
    line_format='user item rating timestamp',
    sep='\t',
    rating_scale=(1,5)
)

data = Dataset.load_from_file(
    "data/ml-100k/u.data",
    reader=reader
)

# -----------------------------------------
# Train Test Split
# -----------------------------------------

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# -----------------------------------------
# Train Model
# -----------------------------------------

model = SVD(random_state=42)

model.fit(trainset)

# -----------------------------------------
# Predictions
# -----------------------------------------

predictions = model.test(testset)

# -----------------------------------------
# Evaluation
# -----------------------------------------

print("="*60)
print("MODEL EVALUATION")
print("="*60)

rmse(predictions)

mae(predictions)