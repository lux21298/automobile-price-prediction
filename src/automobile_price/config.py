from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "imports-85.data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "automobile_price_model.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTION_PLOT_PATH = REPORTS_DIR / "predicted_vs_actual.png"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.png"

TARGET_COLUMN = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2
