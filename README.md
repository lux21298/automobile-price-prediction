# Automobile Price Prediction

A machine learning project for predicting automobile prices using the UCI Automobile dataset. The repository preserves the original DNN-based experiment work while also providing a cleaner, reproducible scikit-learn pipeline and Streamlit demo for portfolio presentation.

## What This Project Shows

- End-to-end tabular regression workflow for car price prediction.
- Data loading and cleaning for mixed numeric and categorical features.
- Original experiment tracking for DNN and wide-and-deep modeling.
- Refactored training pipeline that is easier to run outside Colab.
- Streamlit app for interactive prediction.

## Dataset

This project uses the UCI Automobile dataset.

- Dataset page: https://archive.ics.uci.edu/dataset/10/automobile
- Raw data source: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
- Local copy: `data/raw/imports-85.data`

The dataset contains 205 automobile records with technical and categorical attributes such as make, fuel type, body style, engine size, horsepower, mileage, and price.

## Original Experiment Summary

The original analysis tested multiple modeling strategies and feature transformations before identifying the best-performing setup.

Main methods and parameters preserved from the original work:

- Models: DNN Regressor and DNNLinearCombinedRegressor.
- Best learning rate: `0.005`.
- Best batch size: `16`.
- Hidden layers tested: `128, 64` and `128, 64, 32`.
- Training steps tested: `10,000` to `30,000`.
- Feature transformations: normalization, Min-Max scaling, robust scaling, and log transformation.
- Feature set comparisons: numeric-only, categorical-only, normalized numeric, and all features.

Best original result:

```text
Model: DNNLinearCombinedRegressor
Features: normalized numeric features + encoded categorical features
RMSE: 545.60
R2:   0.9954
```

The detailed original experiment notes are kept in:

```text
experiments/original_dnn_experiments.md
```

## Refactored Baseline Pipeline

The current runnable codebase includes a scikit-learn baseline designed for reproducibility and easy local execution:

- Numeric features: median imputation and standard scaling.
- Categorical features: most-frequent imputation and one-hot encoding.
- Model: RandomForestRegressor.
- Saved artifact: `models/automobile_price_model.pkl`.

This baseline is not meant to replace the original DNN result. It is a clean implementation for demonstration, maintainability, and deployment.

Latest baseline result:

```text
MAE:        1900.23
RMSE:       2907.20
R2:         0.9309
Train rows: 160
Test rows:  41
```

## Project Structure

```text
.
|-- app.py
|-- data/
|   |-- README.md
|   `-- raw/
|       `-- imports-85.data
|-- experiments/
|   `-- original_dnn_experiments.md
|-- reports/
|   `-- .gitkeep
|-- src/
|   `-- automobile_price/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- modeling.py
|       `-- train.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Installation

Create a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

Train the reproducible baseline model:

```bash
python -m automobile_price.train
```

Run the Streamlit demo:

```bash
streamlit run app.py
```

Training generates:

- `models/automobile_price_model.pkl`
- `reports/metrics.json`
- `reports/predicted_vs_actual.png`
- `reports/feature_importance.png`

## Experiment Results

| Experiment | Feature Set | Main Technique | RMSE | R2 |
| --- | --- | --- | ---: | ---: |
| Numeric only | Numeric features | DNN regressor without normalization | 1721.60 | 0.9544 |
| Normalized numeric | Numeric features | DNN regressor with scaling | 1751.49 | 0.9528 |
| Categorical only | Categorical features | Indicator columns | 1741.88 | 0.9553 |
| All features | Numeric + categorical | DNNLinearCombinedRegressor | 545.60 | 0.9954 |
| Refactored baseline | Numeric + categorical | Scikit-learn random forest pipeline | 2907.20 | 0.9309 |

## Next Steps

- Rebuild the best DNNLinearCombinedRegressor experiment in a modern TensorFlow or PyTorch workflow.
- Add cross-validation for more robust evaluation on the small dataset.
- Compare random forest, gradient boosting, and neural-network regressors.
- Deploy the Streamlit demo as a live app.

## References

- UCI Machine Learning Repository. Automobile dataset. https://archive.ics.uci.edu/dataset/10/automobile
- Google Developers. Numerical data: Normalization. https://developers.google.com/machine-learning/crash-course/numerical-data/normalization

