import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 03: Model Baselines\n",
    "Training and evaluating baseline models (Logistic Regression & Random Forest)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 1 — Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "sys.path.append(str(Path().resolve().parent))\n",
    "from src.data.data_loader import (\n",
    "    load_config,\n",
    "    load_dataset,\n",
    "    sample_dataset\n",
    ")\n",
    "from src.data.preprocessing import (\n",
    "    basic_preprocessing_pipeline\n",
    ")\n",
    "from src.features.feature_engineering import (\n",
    "    feature_engineering_pipeline\n",
    ")\n",
    "from src.features.encoder import (\n",
    "    encode_categorical_columns\n",
    ")\n",
    "from src.models.train_test_split import (\n",
    "    temporal_train_test_split,\n",
    "    split_features_target\n",
    ")\n",
    "from src.models.smote_pipeline import (\n",
    "    apply_smote\n",
    ")\n",
    "from src.models.baseline_models import (\n",
    "    train_logistic_regression,\n",
    "    train_random_forest,\n",
    "    generate_predictions\n",
    ")\n",
    "from src.evaluation.metrics import (\n",
    "    evaluate_model,\n",
    "    plot_confusion_matrix\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 2 — Load Config"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "config = load_config()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 3 — Load Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = load_dataset(config)\n",
    "df = sample_dataset(df, config)\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 4 — Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = basic_preprocessing_pipeline(df)\n",
    "df = feature_engineering_pipeline(df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5 — Temporal Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df, test_df = temporal_train_test_split(\n",
    "    df\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 6 — Feature/Target Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = split_features_target(\n",
    "    train_df,\n",
    "    test_df\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 7 — Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, encoders = encode_categorical_columns(\n",
    "    X_train,\n",
    "    X_test\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8 — SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_smote, y_train_smote = apply_smote(\n",
    "    X_train,\n",
    "    y_train\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 9 — Logistic Regression Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "logistic_model = train_logistic_regression(\n",
    "    X_train_smote,\n",
    "    y_train_smote\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 10 — Logistic Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "logistic_preds, logistic_probs = generate_predictions(\n",
    "    logistic_model,\n",
    "    X_test\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 11 — Logistic Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluate_model(\n",
    "    y_test,\n",
    "    logistic_preds,\n",
    "    logistic_probs,\n",
    "    model_name=\"Logistic Regression\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 12 — Logistic Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_confusion_matrix(\n",
    "    y_test,\n",
    "    logistic_preds,\n",
    "    model_name=\"Logistic Regression\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 13 — Random Forest Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_model = train_random_forest(\n",
    "    X_train_smote,\n",
    "    y_train_smote\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 14 — Random Forest Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_preds, rf_probs = generate_predictions(\n",
    "    rf_model,\n",
    "    X_test\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 15 — Random Forest Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "evaluate_model(\n",
    "    y_test,\n",
    "    rf_preds,\n",
    "    rf_probs,\n",
    "    model_name=\"Random Forest\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 16 — Random Forest Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_confusion_matrix(\n",
    "    y_test,\n",
    "    rf_preds,\n",
    "    model_name=\"Random Forest\"\n",
    ")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Save the notebook
with open('notebooks/03_model_baselines.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("✓ Notebook created successfully at notebooks/03_model_baselines.ipynb")
print("\n📋 This notebook requires the following modules to be implemented:")
print("  - src/data/preprocessing.py (basic_preprocessing_pipeline)")
print("  - src/features/feature_engineering.py (feature_engineering_pipeline)")
print("  - src/features/encoder.py (encode_categorical_columns)")
print("  - src/models/train_test_split.py (temporal split functions)")
print("  - src/models/smote_pipeline.py (apply_smote)")
print("  - src/models/baseline_models.py (train_logistic_regression, train_random_forest, generate_predictions)")
print("  - src/evaluation/metrics.py (evaluate_model, plot_confusion_matrix)")
print("\n⚠️  Make sure to create these modules before running the notebook!")