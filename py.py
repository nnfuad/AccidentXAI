import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 06: SHAP Analysis\n",
    "Model interpretability using SHAP (SHapley Additive exPlanations)"
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
    "from src.visualization.shap_visualizer import (\n",
    "    create_tree_explainer,\n",
    "    compute_shap_values,\n",
    "    plot_shap_summary,\n",
    "    plot_shap_dependence,\n",
    "    plot_shap_waterfall\n",
    ")\n",
    "from src.evaluation.feature_importance import (\n",
    "    compute_mean_shap_importance\n",
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
    "## Cell 4 — Create SHAP Sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_shap_sample = X_test.sample(\n",
    "    n=1000,\n",
    "    random_state=42\n",
    ")\n",
    "X_shap_sample.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5 — IMPORTANT RESEARCH INSIGHT\n",
    "\n",
    "> **Never compute SHAP on entire huge datasets initially.**\n",
    "> \n",
    "> SHAP is computationally expensive.\n",
    "> \n",
    "> Sampling is common and acceptable practice."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 6 — Create Explainer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "explainer = create_tree_explainer(\n",
    "    xgb_model\n",
    ")\n",
    "\n",
    "print(\"✓ SHAP TreeExplainer created\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 7 — Compute SHAP Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "shap_values = compute_shap_values(\n",
    "    explainer,\n",
    "    X_shap_sample\n",
    ")\n",
    "\n",
    "print(\"✓ SHAP values computed successfully\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8 — SHAP Summary Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_shap_summary(\n",
    "    shap_values,\n",
    "    X_shap_sample\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 9 — What You Should Observe\n",
    "\n",
    "**Features at top are most influential.**\n",
    "\n",
    "| Color | Meaning |\n",
    "|-------|---------|\n",
    "| Red | high feature value |\n",
    "| Blue | low feature value |\n",
    "\n",
    "**Horizontal spread shows impact magnitude.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 10 — Feature Importance Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "importance_df = compute_mean_shap_importance(\n",
    "    shap_values,\n",
    "    X_shap_sample\n",
    ")\n",
    "importance_df.head(15)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 11 — Dependence Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_shap_dependence(\n",
    "    shap_values,\n",
    "    X_shap_sample,\n",
    "    feature_name=\"Visibility(mi)\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 12 — What Dependence Plot Means\n",
    "\n",
    "You are analyzing:\n",
    "\n",
    "**How changing visibility changes model prediction**\n",
    "\n",
    "This becomes scientifically interesting.\n",
    "\n",
    "---\n",
    "\n",
    "**Interpretation Guide:**\n",
    "- X-axis: Feature value (Visibility in miles)\n",
    "- Y-axis: SHAP value (Impact on prediction)\n",
    "- Positive SHAP: Increases accident severity\n",
    "- Negative SHAP: Decreases accident severity\n",
    "- Color (if shown): Secondary feature interaction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 13 — Local Explanation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_shap_waterfall(\n",
    "    explainer,\n",
    "    shap_values,\n",
    "    X_shap_sample,\n",
    "    sample_index=0\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optional Cell 14 — Additional SHAP Analysis\n",
    "\n",
    "Explore more SHAP visualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bar plot of feature importance\n",
    "import matplotlib.pyplot as plt\n",
    "import shap\n",
    "\n",
    "# Summary bar plot\n",
    "shap.summary_plot(shap_values, X_shap_sample, plot_type=\"bar\")\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Force plot for first prediction\n",
    "shap.initjs()\n",
    "shap.force_plot(explainer.expected_value, shap_values[0,:], X_shap_sample.iloc[0,:], matplotlib=True)\n",
    "plt.show()"
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
with open('notebooks/06_shap_analysis.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("✓ Notebook created successfully at notebooks/06_shap_analysis.ipynb")
print("\n📋 This notebook requires the following modules to be implemented:")
print("  - src/data/preprocessing.py (basic_preprocessing_pipeline)")
print("  - src/features/feature_engineering.py (feature_engineering_pipeline)")
print("  - src/features/encoder.py (encode_categorical_columns)")
print("  - src/models/train_test_split.py (temporal split functions)")
print("  - src/models/smote_pipeline.py (apply_smote)")
print("  - src/models/boosting_models.py (train_xgboost)")
print("  - src/visualization/shap_visualizer.py (SHAP visualization functions)")
print("  - src/evaluation/feature_importance.py (compute_mean_shap_importance)")
print("\n⚠️  Make sure to install required packages:")
print("  pip install shap xgboost")
print("\n⚠️  Make sure to create these modules before running the notebook!")