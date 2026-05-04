import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_task1_dataset(df: pd.DataFrame):
    """
    Build features and target for Task 1 (career outcome prediction).
    """

    df = df.copy()
    df = df.dropna(subset=["Future Career"])

    skill_map = {"Weak": 1, "Moderate": 2, "Strong": 3}
    df["Python"] = df["Python"].map(skill_map)
    df["SQL"] = df["SQL"].map(skill_map)
    df["Java"] = df["Java"].map(skill_map)

    # Setting the target
    y = df["Future Career"]

    # Selecting the features
    feature_cols = [
        "Gender",
        "Age",
        "GPA",
        "Major",
        "Interested Domain",
        "Projects",
        "Python",
        "SQL",
        "Java",
    ]

    X = df[feature_cols]
    
    categorical_cols = ["Gender", "Major", "Interested Domain", "Projects"]
    numeric_cols = ["Age", "GPA", "Python", "SQL", "Java"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
