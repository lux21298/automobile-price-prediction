from __future__ import annotations

import json
import os
import pickle

from sklearn.model_selection import train_test_split

from .config import (
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREDICTION_PLOT_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from .data import load_modeling_data
from .modeling import build_pipeline, evaluate_regression, get_feature_importance

os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_prediction_plot(y_true, y_pred) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.75)
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=2)
    plt.title("Predicted vs Actual Automobile Prices")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=160)
    plt.close()


def save_feature_importance_plot(model) -> None:
    importance = get_feature_importance(model).head(15)
    plt.figure(figsize=(8, 6))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=160)
    plt.close()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_modeling_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = evaluate_regression(y_test, predictions)
    metrics["train_rows"] = int(len(X_train))
    metrics["test_rows"] = int(len(X_test))

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_prediction_plot(y_test, predictions)
    save_feature_importance_plot(model)

    print("Training complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
