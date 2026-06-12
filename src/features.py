import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    )

    return df