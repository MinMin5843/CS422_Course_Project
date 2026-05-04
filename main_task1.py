import pandas as pd
from src.task1_preprocess import build_task1_dataset
from src.task1_model import train_task1_models

def main():
    df = pd.read_csv("data/cs_students.csv")

    X_train, X_test, y_train, y_test, preprocessor = build_task1_dataset(df)

    train_task1_models(X_train, X_test, y_train, y_test, preprocessor)


if __name__ == "__main__":
    main()