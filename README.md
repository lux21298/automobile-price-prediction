# Automobile Price Prediction

A reproducible machine learning project for predicting automobile prices using the UCI Automobile dataset. This repository includes data ingestion, preprocessing, training, evaluation, and a Streamlit demo for interactive prediction.

## What’s included

- `data/raw/imports-85.data`: local copy of the UCI Automobile dataset.
- `src/automobile_price/`: project package with data loading, preprocessing, modeling, and training logic.
- `app.py`: Streamlit application for exploring the model and predicting car prices.
- `reports/`: generated reports and metrics artifacts.
- `models/`: saved model artifact output.

## Installation

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

Train the model:

```bash
python -m automobile_price.train
```

Run the demo:

```bash
streamlit run app.py
```

## Project structure

```text
.
├── app.py
├── data/
│   ├── README.md
│   └── raw/
│       └── imports-85.data
├── models/
├── reports/
│   └── .gitkeep
├── requirements.txt
├── src/
│   └── automobile_price/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── modeling.py
│       └── train.py
└── README.md
```

## Dataset

This project uses the UCI Automobile dataset:

- Dataset page: https://archive.ics.uci.edu/dataset/10/automobile
- Raw data file: `data/raw/imports-85.data`

## Model pipeline

The implemented pipeline handles mixed numeric and categorical features:

- numeric features: missing value handling, scaling
- categorical features: imputation and one-hot encoding
- regression model: scikit-learn estimator

## Results

Training produces evaluation metrics and artifacts in `reports/` and a saved model in `models/`.

Example metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

## Recommendations

- Add additional model comparison (e.g. gradient boosting, neural networks).
- Add cross-validation for more robust evaluation.
- Deploy the Streamlit app as a live demo.

## License

This repository is provided as a portfolio project.
