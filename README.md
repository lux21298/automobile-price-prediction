# Automobile Price Prediction

A compact machine learning project that predicts automobile prices from technical and categorical vehicle attributes. The project turns the original exploratory Colab exercise into a reproducible portfolio-ready pipeline with local data loading, preprocessing, model training, evaluation, and a small Streamlit demo.

## Project Highlights

- Uses the UCI Automobile dataset with 205 vehicle records and 26 attributes.
- Cleans missing values marked as `?` and converts numeric columns safely.
- Trains a scikit-learn regression pipeline with preprocessing built in.
- Saves model artifacts, evaluation metrics, and diagnostic charts.
- Includes a Streamlit app for interactive price prediction.

## Dataset

The dataset is the UCI Automobile dataset.

- Source page: https://archive.ics.uci.edu/dataset/10/automobile
- Raw data URL: https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
- Local copy: `data/raw/imports-85.data`

The original Colab notebook referenced a Google-hosted copy of the same car dataset, but the stable source for a portfolio project is the UCI repository.

## Project Structure

```text
.
|-- app.py
|-- data/
|   |-- README.md
|   `-- raw/
|       `-- imports-85.data
|-- reports/
|   `-- .gitkeep
|-- src/
|   `-- automobile_price/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- modeling.py
|       `-- train.py
|-- requirements.txt
`-- README.md
```

## Quick Start

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Train the model:

```bash
python -m automobile_price.train
```

Run the Streamlit demo:

```bash
streamlit run app.py
```

## Outputs

Training creates:

- `models/automobile_price_model.pkl`
- `reports/metrics.json`
- `reports/predicted_vs_actual.png`
- `reports/feature_importance.png`

## Modeling Approach

The model predicts `price` using a preprocessing and regression pipeline:

- Numeric columns: median imputation and standard scaling.
- Categorical columns: most-frequent imputation and one-hot encoding.
- Regressor: random forest regression.

This keeps preprocessing and prediction together in one saved artifact, which reduces the chance of training-serving mismatch.

## Portfolio Notes

For recruiters, this project demonstrates:

- Translating a notebook exercise into maintainable project code.
- Building reproducible data and model pipelines.
- Handling mixed numeric and categorical tabular data.
- Producing clear evaluation artifacts and an interactive demo.
