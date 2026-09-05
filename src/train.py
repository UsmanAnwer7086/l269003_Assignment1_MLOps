# ============================================================
# MLOps Assignment 1
# Student ID: 26L-9003
# House Price Prediction using Random Forest Regression
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# 1. STUDENT INFORMATION
# ------------------------------------------------------------

STUDENT_ID = "26L-9003"

print("=" * 60)
print("MLOps Assignment 1")
print(f"Student ID: {STUDENT_ID}")
print("House Price Prediction")
print("Random Forest Regression")
print("=" * 60)


# ------------------------------------------------------------
# 2. FILE PATHS
# ------------------------------------------------------------

DATA_PATH = "data/housepr_train.csv"

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    f"random_forest_{STUDENT_ID}.pkl")


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nFirst five rows:")
print(df.head())


# ------------------------------------------------------------
# 4. REMOVE UNNECESSARY COLUMN
# ------------------------------------------------------------

# Id is only an identifier and is not used as a predictive feature.

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

print("\nShape after removing Id:")
print(df.shape)


# ------------------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

TARGET_COLUMN = "SalePrice"

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

print("\nNumber of features:", X.shape[1])
print("Target column:", TARGET_COLUMN)


# ------------------------------------------------------------
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumber of numerical features:",
      len(numerical_features))

print("Number of categorical features:",
      len(categorical_features))


# ------------------------------------------------------------
# 7. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

### FILLING THE MISSING VALUES USING IMPUTER
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ------------------------------------------------------------
# 8. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------------------------
# 9. COMBINE PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------------------------
# 10. TRAIN-TEST SPLIT
# ------------------------------------------------------------
## SPLITTING THE DATA INTO TRAIN-TEST Sub-PARTS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 11. RANDOM FOREST REGRESSOR
# ------------------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=101,
    n_jobs=-1
)


# ------------------------------------------------------------
# 12. COMPLETE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor,
        ),
        (
            "random_forest",
            random_forest
        )
    ]
)


# ------------------------------------------------------------
# 13. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest model...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ------------------------------------------------------------
# 14. PREDICTION
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = model_pipeline.predict(X_test)


# ------------------------------------------------------------
# 15. MODEL EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE  : {mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ------------------------------------------------------------
# 16. CREATE MODEL DIRECTORY
# ------------------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# 17. SAVE TRAINED MODEL
# ------------------------------------------------------------

joblib.dump(
    model_pipeline,
    MODEL_PATH
)

print("\nModel saved successfully.")

print(f"Model path: {MODEL_PATH}")


