# Fashion-MNIST ANN Pipeline with Git & DVC

An end-to-end Machine Learning pipeline utilizing Git for code versioning, Data Version Control (DVC) with Google Drive as remote storage for data and model versioning, and TensorFlow for training a fully-connected Artificial Neural Network (ANN) on the Fashion-MNIST dataset.

## Target
- Achieve $\ge 85\%$ test accuracy on Fashion-MNIST using a Sequential ANN architecture.
- Full pipeline reproducibility via `dvc repro`.

## Project Structure
- `src/prepare.py` - Downloads Fashion-MNIST and saves raw data to `data/raw/`
- `src/preprocess.py` - Normalizes features and creates train/val splits in `data/processed/`
- `src/train.py` - Trains the Sequential ANN using hyperparameters from `params.yaml`
- `src/evaluate.py` - Evaluates model performance, outputs confusion matrix and `metrics.json`
- `params.yaml` - Central configuration file for hyperparameters
- `dvc.yaml` - Pipeline stage definitions and dependency graph

