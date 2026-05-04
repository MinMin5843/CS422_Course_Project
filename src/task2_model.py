from typing import Dict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.utils import print_section


def _build_models() -> Dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def train_task2_models(X_train, X_test, y_train, y_test, preprocessor):
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

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
