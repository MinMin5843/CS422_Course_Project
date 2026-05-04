import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def compute_difficulty_score(row: pd.Series) -> float:
    """
    Calculate a difficulty scoring logic based on the:
      1. Number of skills required
      2. Experience required
      3. Salary competitiveness
      4. Company size (larger companies often have harder/longer interview processes)
    """

    score = 0.0

    # More skills = more competitive
    score += 1.2 * row["num_skills"]

    # More years of experience = more competitive
    score += 2.0 * row["experience_required"]

    # Higher salary = more competitive
    score += 0.0005 * row["salary_avg"]

    # Larger companies = more competitive (startups)
    company_size_map = {
        "1-50 employees": 0,
        "51-200 employees": 1,
        "201-500 employees": 2,
        "500+ employees": 3
    }
    score += company_size_map.get(row["company_size"], 1)

    return score


def build_task2_dataset(df: pd.DataFrame):
    """
    Build features and target for Task 2 (job posting competitiveness).
    """

    df = df.copy()

    df["salary_avg"] = (df["salary_min"] + df["salary_max"]) / 2

    # Count number of skills
    df["num_skills"] = df["skills"].apply(
        lambda x: len([s.strip() for s in x.split(",")]) if isinstance(x, str) else 0
    )

    # Clean company size
    df["company_size"] = df["company_size"].fillna("Unknown")

    df["difficulty_score"] = df.apply(compute_difficulty_score, axis=1)
    df["difficulty_cat"] = pd.qcut(
        df["difficulty_score"],
        q=3,
        labels=["Low", "Medium", "High"]
    )
    df = df.dropna(subset=["difficulty_cat"])
    
    # Selecting the features

    feature_cols = [
        "job_title",
        "location",
        "job_type",
        "category",
        "education_level",
        "company_size",
        "salary_avg",
        "experience_required",
        "num_skills",
    ]

    X = df[feature_cols]
    y = df["difficulty_cat"]

    categorical_cols = [
        "job_title",
        "location",
        "job_type",
        "category",
        "education_level",
        "company_size",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",  # numeric columns pass through
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
