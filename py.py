import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 07: Temporal SHAP Analysis\n",
    "Analyzing how feature importance changes over time (feature drift analysis)"
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
    "    train_xgboost\n",
    ")\n",
    "from src.visualization.temporal_shap import (\n",
    "    compute_yearly_shap_importance,\n",
    "    plot_temporal_feature_drift,\n",
    "    get_top_features_by_year\n",
    ")\n",
    "from src.evaluation.drift_analysis import (\n",
    "    compute_feature_drift\n",
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
    "print(f\"✓ Data prepared successfully!\")\n",
    "print(f\"  Training set (SMOTE): {X_train_smote.shape}\")\n",
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
    ")\n",
    "\n",
    "print(\"✓ XGBoost model trained successfully\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 4 — Feature Columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_columns = X_train.columns.tolist()\n",
    "feature_columns[:10]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5 — Compute Yearly SHAP Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "yearly_shap_df = compute_yearly_shap_importance(\n",
    "    model=xgb_model,\n",
    "    df=df,\n",
    "    feature_columns=feature_columns,\n",
    "    sample_size=300\n",
    ")\n",
    "yearly_shap_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 6 — IMPORTANT RESEARCH INSIGHT\n",
    "\n",
    "> **We use smaller yearly SHAP samples because:**\n",
    "> \n",
    "> Temporal SHAP is computationally expensive.\n",
    "> \n",
    "> Research practicality matters.\n",
    "> \n",
    "> *\"Better to have approximate answers than exact but late ones.\"*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 7 — Plot Temporal Feature Drift"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_temporal_feature_drift(\n",
    "    yearly_shap_df,\n",
    "    top_n=5\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8 — What You Should Observe\n",
    "\n",
    "**Some features may remain stable.**\n",
    "\n",
    "**Others may drift significantly.**\n",
    "\n",
    "| Feature | Behavior |\n",
    "|---------|----------|\n",
    "| Night Driving | stable |\n",
    "| Visibility | increasing |\n",
    "| Weather | fluctuating |\n",
    "\n",
    "---\n",
    "\n",
    "**This becomes publishable insight.**\n",
    "\n",
    "Understanding *why* features drift can lead to:\n",
    "- Better model retraining strategies\n",
    "- Domain-specific insights about changing conditions\n",
    "- Improved feature engineering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 9 — Drift Analysis Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "drift_df = compute_feature_drift(\n",
    "    yearly_shap_df\n",
    ")\n",
    "drift_df.head(15)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 10 — Top Features By Year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "top_features = get_top_features_by_year(\n",
    "    yearly_shap_df,\n",
    "    top_n=5\n",
    ")\n",
    "top_features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optional Cell 11 — Drift Visualization Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Pivot table for heatmap\n",
    "pivot_df = yearly_shap_df.pivot(\n",
    "    index='feature', \n",
    "    columns='year', \n",
    "    values='mean_shap'\n",
    ")\n",
    "\n",
    "# Plot heatmap\n",
    "plt.figure(figsize=(12, 8))\n",
    "sns.heatmap(pivot_df.head(10), annot=True, fmt='.3f', cmap='RdBu_r', center=0)\n",
    "plt.title('Feature Importance Heatmap Over Time (Top 10 Features)', \n",
    "          fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Year')\n",
    "plt.ylabel('Feature')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optional Cell 12 — Stability Score Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate stability score (inverse of coefficient of variation)\n",
    "stability_df = yearly_shap_df.groupby('feature')['mean_shap'].agg(['mean', 'std'])\n",
    "stability_df['cv'] = stability_df['std'] / stability_df['mean']\n",
    "stability_df['stability_score'] = 1 / stability_df['cv']\n",
    "stability_df = stability_df.sort_values('stability_score', ascending=False)\n",
    "\n",
    "print(\"Feature Stability Scores (higher = more stable):\")\n",
    "print(stability_df[['mean', 'std', 'stability_score']].head(10))\n",
    "\n",
    "print(\"\\n\\nMost Unstable Features (potential drift):\")\n",
    "print(stability_df[['mean', 'std', 'stability_score']].tail(10))"
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
with open('notebooks/07_temporal_shap_analysis.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("✓ Notebook created successfully at notebooks/07_temporal_shap_analysis.ipynb")
print("\n📋 This notebook requires the following modules to be implemented:")
print("  - src/data/preprocessing.py (basic_preprocessing_pipeline)")
print("  - src/features/feature_engineering.py (feature_engineering_pipeline)")
print("  - src/features/encoder.py (encode_categorical_columns)")
print("  - src/models/train_test_split.py (temporal split functions)")
print("  - src/models/smote_pipeline.py (apply_smote)")
print("  - src/models/boosting_models.py (train_xgboost)")
print("  - src/visualization/temporal_shap.py (temporal SHAP functions)")
print("  - src/evaluation/drift_analysis.py (compute_feature_drift)")
print("\n⚠️  Make sure to install required packages:")
print("  pip install shap xgboost pandas matplotlib seaborn")
print("\n⚠️  Make sure to create these modules before running the notebook!")