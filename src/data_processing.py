import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "churn.csv"
)

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["customerID", "Churn"])
Y = df["Churn"]

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

num_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols
        ),
        (
            "num",
            StandardScaler(),
            num_cols
        )
    ]
)

# define get_processor function
def get_processor():
    return preprocessor

# define load_data function
def load_data():
    return X,Y