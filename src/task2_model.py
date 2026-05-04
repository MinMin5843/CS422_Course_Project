from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.utils import print_section


def _build_models() -> Dict[str, object]:
    """
    Define the models to be tested for Task 2.
    Includes both linear and tree-based models.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            multi_class="auto"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }
    return models


def train_task2_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate multiple models for Task 2 using a shared preprocessor.
    """
    models = _build_models()

    for name, model in models.items():
        print_section(f"Task 2 - Training {name}")

        clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))
