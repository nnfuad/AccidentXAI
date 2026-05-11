import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 04: Boosting Models\n",
    "Training and evaluating gradient boosting models (XGBoost, LightGBM, CatBoost)"
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
    "from src.models.boosting_models import (\n",
    "    train_xgboost,\n",
    "    train_lightgbm,\n",
    "    train_catboost\n",
    ")\n",
    "from src.models.baseline_models import (\n",
    "    generate_predictions\n",
    ")\n",
    "from src.evaluation.metrics import (\n",
    "    evaluate_model,\n",
    "    plot_confusion_matrix\n",
    ")\n",
    "from src.evaluation.cross_validation import (\n",
    "    run_cross_validation\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 2 — Load + Prepare Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "config = load_config()\n",
    "df = load_dataset(config)\n",
    "df = sample_dataset(df, config)\n",
    "df = basic_preprocessing_pipeline(df)\n",
    "df = feature_engineering_pipeline(df)\n",
    "train_df, test_df = temporal_train_test_split(df)\n",
    "X_train, X_test, y_train, y_test = split_features_target(\n",
    "    train_df,\n",
    "    test_df\n",
    ")\n",
    "X_train, X_test, encoders = encode_categorical_columns(\n",
    "    X_train,\n",
    "    X_test\n",
    ")\n",
    "X_train_smote, y_train_smote = apply_smote(\n",
    "    X_train,\n",
    "    y_train\n",
    ")\n",
    "\n",
    "print(f\"\\n✓ Data prepared successfully!\")\n",
    "print(f\"  Training set: {X_train_smote.shape}\")\n",
    "print(f\"  Test set: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 3 — Train XGBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_model = train_xgboost(\n",
    "    X_train_smote,\n",
    "    y_train_smote\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 4 — XGBoost Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_preds, xgb_probs = generate_predictions(\n",
    "    xgb_model,\n",
    "    X_test\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5 — XGBoost Evaluation"
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
    "    xgb_preds,\n",
    "    xgb_probs,\n",
    "    model_name=\"XGBoost\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 6 — XGBoost Confusion Matrix"
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
    "    xgb_preds,\n",
    "    model_name=\"XGBoost\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 7 — XGBoost Cross Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "run_cross_validation(\n",
    "    xgb_model,\n",
    "    X_train_smote,\n",
    "    y_train_smote\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8 — Train LightGBM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "lgbm_model = train_lightgbm(\n",
    "    X_train_smote,\n",
    "    y_train_smote\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 9 — LightGBM Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "lgbm_preds, lgbm_probs = generate_predictions(\n",
    "    lgbm_model,\n",
    "    X_test\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 10 — LightGBM Evaluation"
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
    "    lgbm_preds,\n",
    "    lgbm_probs,\n",
    "    model_name=\"LightGBM\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 11 — Train CatBoost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_model = train_catboost(\n",
    "    X_train_smote,\n",
    "    y_train_smote\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 12 — CatBoost Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_preds, cat_probs = generate_predictions(\n",
    "    cat_model,\n",
    "    X_test\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 13 — CatBoost Evaluation"
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
    "    cat_preds,\n",
    "    cat_probs,\n",
    "    model_name=\"CatBoost\"\n",
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
with open('notebooks/04_boosting_models.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("✓ Notebook created successfully at notebooks/04_boosting_models.ipynb")
print("\n📋 This notebook requires the following modules to be implemented:")
print("  - src/data/preprocessing.py (basic_preprocessing_pipeline)")
print("  - src/features/feature_engineering.py (feature_engineering_pipeline)")
print("  - src/features/encoder.py (encode_categorical_columns)")
print("  - src/models/train_test_split.py (temporal split functions)")
print("  - src/models/smote_pipeline.py (apply_smote)")
print("  - src/models/boosting_models.py (train_xgboost, train_lightgbm, train_catboost)")
print("  - src/models/baseline_models.py (generate_predictions)")
print("  - src/evaluation/metrics.py (evaluate_model, plot_confusion_matrix)")
print("  - src/evaluation/cross_validation.py (run_cross_validation)")
print("\n⚠️  Make sure to install required packages:")
print("  pip install xgboost lightgbm catboost scikit-learn imbalanced-learn")
print("\n⚠️  Make sure to create these modules before running the notebook!")