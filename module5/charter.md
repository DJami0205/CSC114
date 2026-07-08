# Project Charter: Texas Home Price Predictor

## What we're building (one sentence)
A model that predicts home price for a Texas property from its listing features.

## Cohort
Tabular

## The data or tools we'll use
- **Dataset:** Texas Residential Real Estate Intelligence 2026 (Kaggle)
- **Tools:** Keras; k-fold validation

## Definition of "good enough"
Before we build, we agree this project is good enough when:
- Validation MAE (Mean Absolute Error, in dollars) beats the naive (predict-the-mean) baseline.
- Validation MAE stops improving (the turnaround epoch).

## What we are NOT doing (scope guard)
- Feature engineering beyond what's provided in the dataset.
- Deployment (no live app or website).

## Team & roles
Solo. Self-review documented in each pull request.
