# Original DNN Experiment Notes

This file preserves the original modeling approach, hyperparameters, feature engineering choices, and evaluation results from the initial automobile price prediction work. It is kept as an experiment record, while the main `src/` package provides a cleaner reproducible baseline for local execution and demo purposes.

## Objective

Predict automobile prices using the UCI Automobile dataset by comparing model architectures, feature sets, and feature transformations.

## Models Tested

### DNN Regressor

The DNN Regressor was used as the main baseline model for numeric and transformed numeric feature experiments.

Tested configurations included:

- Hidden layers: `64`
- Hidden layers: `128, 64`
- Hidden layers: `128, 64, 32`
- Batch sizes: `10`, `16`, `32`
- Learning rates: `0.1` to `0.0005`
- Training steps: `10,000` to `30,000`

Best observed settings:

```text
Learning rate: 0.005
Batch size:    16
Hidden layers: 128, 64, 32
```

### DNNLinearCombinedRegressor

The best-performing original model used a combined linear and deep architecture:

- Linear component for categorical indicators.
- Deep component for numeric features.
- Mixed feature input to capture both linear category effects and non-linear numeric relationships.

This approach performed best because the dataset contains both continuous measurements and categorical descriptors.

## Feature Engineering

Several transformations were tested to improve learning stability and model quality.

### Z-score Normalization

Applied to scale-sensitive continuous features such as:

- `width`
- `height`
- `curb-weight`
- `length`
- `highway-mpg`

### Min-Max Scaling

Applied to bounded or smaller-range features such as:

- `symboling`
- `wheel-base`

### Robust Scaling

Used for features where median and interquartile range were more suitable than mean and standard deviation:

- `city-mpg`
- `stroke`
- `normalized-losses`
- `engine-size`
- `horsepower`

### Log Transformation

Applied to skewed variables:

- `peak-rpm`
- `compression-ratio`

## Experiment Results

| Model | Feature Set | Notes | RMSE | R2 |
| --- | --- | --- | ---: | ---: |
| Model 1 | Numeric only | DNN Regressor without normalization | 1721.60 | 0.9544 |
| Model 2 | Normalized numeric | DNN Regressor with Z-score, Min-Max, robust scaling, and log transforms | 1751.49 | 0.9528 |
| Model 3 | Categorical only | Encoded categorical indicator features | 1741.88 | 0.9553 |
| Model 4 | Numeric + categorical | DNNLinearCombinedRegressor with transformed numeric and encoded categorical features | 545.60 | 0.9954 |

## Best Original Result

```text
Model: DNNLinearCombinedRegressor
Feature set: normalized numeric features + encoded categorical features
Learning rate: 0.005
Batch size: 16
Hidden layers: 128, 64, 32
Training steps tested: 10,000 to 30,000
RMSE: 545.60
R2: 0.9954
```

## Interpretation

The combined model performed best because it could learn from both sides of the dataset:

- Numeric features such as engine size, horsepower, curb weight, and mileage explain continuous price variation.
- Categorical features such as make, body style, drive wheels, and fuel system capture market and configuration differences.

The experiment also showed that feature transformations are not automatically beneficial unless they are matched to the feature distribution and followed by hyperparameter retuning.

## Relationship To The Refactored Project

The main project code now uses a scikit-learn random forest pipeline for a clean, reproducible baseline that is easy to run locally and demonstrate in Streamlit.

The original DNN result is preserved here as the primary experiment history, while the refactored baseline provides a maintainable implementation for portfolio review.

