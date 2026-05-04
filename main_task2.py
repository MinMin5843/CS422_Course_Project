import pandas as pd
from src.task2_preprocess import build_task2_dataset
from src.task2_model import train_task2_models


def main():
    df = pd.read_csv("data/job_market.csv")

    X_train, X_test, y_train, y_test, preprocessor = build_task2_dataset(df)

    train_task2_models(X_train, X_test, y_train, y_test, preprocessor)


if __name__ == "__main__":
    main()
