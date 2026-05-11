import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 02: Preprocessing & Feature Engineering\n",
    "Data preprocessing, feature engineering, and train-test preparation"
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
    "import pandas as pd\n",
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
    "from src.models.train_test_split import (\n",
    "    temporal_train_test_split,\n",
    "    split_features_target,\n",
    "    check_class_distribution\n",
    ")\n",
    "from src.models.smote_pipeline import (\n",
    "    apply_smote\n",
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
    "config = load_config()\n",
    "config"
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
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 4 — Sample Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = sample_dataset(df, config)\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5 — Run Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = basic_preprocessing_pipeline(df)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 6 — Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = feature_engineering_pipeline(df)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 7 — Check Engineered Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns.tolist()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8 — Temporal Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df, test_df = temporal_train_test_split(\n",
    "    df,\n",
    "    train_end_year=2021,\n",
    "    test_year=2022\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 9 — Feature/Target Split"
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
    "## Cell 10 — Class Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "check_class_distribution(\n",
    "    y_train,\n",
    "    y_test\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 11 — Apply SMOTE"
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
    "## Cell 12 — Verify Shapes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"X_train_smote shape: {X_train_smote.shape}\")\n",
    "print(f\"y_train_smote shape: {y_train_smote.shape}\")\n",
    "print(f\"\\nOriginal X_train shape: {X_train.shape}\")\n",
    "print(f\"SMOTE increased samples by: {X_train_smote.shape[0] - X_train.shape[0]} ({((X_train_smote.shape[0] - X_train.shape[0])/X_train.shape[0])*100:.1f}%)\")"
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
with open('notebooks/02_preprocessing.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("✓ Notebook created successfully at notebooks/02_preprocessing.ipynb")
print("\n📋 This notebook requires the following modules to be implemented:")
print("  - src/data/preprocessing.py (basic_preprocessing_pipeline)")
print("  - src/features/feature_engineering.py (feature_engineering_pipeline)")
print("  - src/models/train_test_split.py (temporal split functions)")
print("  - src/models/smote_pipeline.py (apply_smote)")
print("\n⚠️  Make sure to create these modules before running the notebook!")