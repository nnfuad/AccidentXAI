import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook 08: Final Pipeline & Model Comparison\n",
    "Complete model training, evaluation, comparison, and interpretability analysis"
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
    "    train_logistic_regression,\n",
    "    train_random_forest,\n",
    "    generate_predictions\n",
    ")\n",
    "from src.evaluation.model_comparison import (\n",
    "    ModelComparison\n",
    ")\n",
    "from src.visualization.shap_visualizer import (\n",
    "    create_tree_explainer,\n",
    "    compute_shap_values,\n",
    "    plot_shap_summary\n",
    ")\n",
    "from src.visualization.temporal_shap import (\n",
    "    compute_yearly_shap_importance,\n",
    "    plot_temporal_feature_drift\n",
    ")\n",
    "from src.visualization.save_figures import (\n",
    "    save_current_figure\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 2 — Load and Prepare Dataset"
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
    "print(f\"  Test set: {X_test.shape}\")\n",
    "print(f\"  Features: {X_train.shape[1]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 3 — Initialize Comparison Engine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison_engine = ModelComparison()\n",
    "print(\"✓ Model comparison engine initialized\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 4 — Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "log_model = train_logistic_regression(\n",
    "    X_train_smote,\n",
    "    y_train_smote\n",
    ")\n",
    "log_preds, log_probs = generate_predictions(\n",
    "    log_model,\n",
    "    X_test\n",
    ")\n",
    "comparison_engine.add_model_result(\n",
    "    \"Logistic Regression\",\n",
    "    y_test,\n",
    "    log_preds,\n",
    "    log_probs\n",
    ")\n",
    "print(\"✓ Logistic Regression completed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5 — Random Forest"
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
    ")\n",
    "rf_preds, rf_probs = generate_predictions(\n",
    "    rf_model,\n",
    "    X_test\n",
    ")\n",
    "comparison_engine.add_model_result(\n",
    "    \"Random Forest\",\n",
    "    y_test,\n",
    "    rf_preds,\n",
    "    rf_probs\n",
    ")\n",
    "print(\"✓ Random Forest completed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 6 — XGBoost"
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
    "xgb_preds, xgb_probs = generate_predictions(\n",
    "    xgb_model,\n",
    "    X_test\n",
    ")\n",
    "comparison_engine.add_model_result(\n",
    "    \"XGBoost\",\n",
    "    y_test,\n",
    "    xgb_preds,\n",
    "    xgb_probs\n",
    ")\n",
    "print(\"✓ XGBoost completed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 7 — LightGBM"
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
    ")\n",
    "lgbm_preds, lgbm_probs = generate_predictions(\n",
    "    lgbm_model,\n",
    "    X_test\n",
    ")\n",
    "comparison_engine.add_model_result(\n",
    "    \"LightGBM\",\n",
    "    y_test,\n",
    "    lgbm_preds,\n",
    "    lgbm_probs\n",
    ")\n",
    "print(\"✓ LightGBM completed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 8 — CatBoost"
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
    ")\n",
    "cat_preds, cat_probs = generate_predictions(\n",
    "    cat_model,\n",
    "    X_test\n",
    ")\n",
    "comparison_engine.add_model_result(\n",
    "    \"CatBoost\",\n",
    "    y_test,\n",
    "    cat_preds,\n",
    "    cat_probs\n",
    ")\n",
    "print(\"✓ CatBoost completed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 9 — Comparison Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison_df = comparison_engine.get_comparison_table()\n",
    "comparison_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 10 — Save Comparison Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "comparison_engine.save_results()\n",
    "print(\"✓ Model comparison results saved to reports/tables/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 11 — SHAP Analysis"
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
    "explainer = create_tree_explainer(\n",
    "    xgb_model\n",
    ")\n",
    "shap_explanation, shap_values = compute_shap_values(\n",
    "    explainer,\n",
    "    X_shap_sample\n",
    ")\n",
    "print(\"✓ SHAP analysis completed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 12 — SHAP Summary Plot"
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
    ")\n",
    "save_current_figure(\n",
    "    \"shap_summary_plot.png\"\n",
    ")\n",
    "print(\"✓ SHAP summary plot saved to reports/figures/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 13 — Temporal SHAP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_columns = X_train.columns.tolist()\n",
    "yearly_shap_df = compute_yearly_shap_importance(\n",
    "    model=xgb_model,\n",
    "    df=df,\n",
    "    feature_columns=feature_columns,\n",
    "    encoders=encoders,\n",
    "    sample_size=300\n",
    ")\n",
    "print(\"✓ Temporal SHAP analysis completed\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 14 — Temporal Drift Plot"
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
    ")\n",
    "save_current_figure(\n",
    "    \"temporal_shap_drift.png\"\n",
    ")\n",
    "print(\"✓ Temporal drift plot saved to reports/figures/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 15 — Save Temporal SHAP Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "yearly_shap_df.to_csv(\n",
    "    \"reports/tables/yearly_shap_importance.csv\",\n",
    "    index=False\n",
    ")\n",
    "print(\"Temporal SHAP table saved successfully to reports/tables/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optional Cell 16 — Generate Final Report Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print final summary\n",
    "print(\"\\n\" + \"=\"*60)\n",
    "print(\"FINAL PIPELINE EXECUTION SUMMARY\")\n",
    "print(\"=\"*60)\n",
    "\n",
    "# Best model\n",
    "best_model = comparison_df.iloc[0]['Model']\n",
    "best_f1 = comparison_df.iloc[0]['F1 Score (Weighted)']\n",
    "print(f\"\\n🏆 Best Model: {best_model}\")\n",
    "print(f\"   Weighted F1 Score: {best_f1:.4f}\")\n",
    "\n",
    "# Top 3 features from SHAP\n",
    "print(f\"\\n📊 Top 3 Most Important Features (from SHAP):\")\n",
    "top_features = compute_mean_shap_importance(shap_values, X_shap_sample)\n",
    "for i, row in top_features.head(3).iterrows():\n",
    "    print(f\"   {i+1}. {row['Feature']}: {row['Mean |SHAP Value|']:.4f}\")\n",
    "\n",
    "# Temporal drift insights\n",
    "print(f\"\\n⏰ Temporal Drift Insights:\")\n",
    "from src.evaluation.drift_analysis import compute_feature_drift\n",
    "drift_df = compute_feature_drift(yearly_shap_df)\n",
    "increasing = drift_df[drift_df['drift_category'] == 'increasing'].head(3)\n",
    "for _, row in increasing.iterrows():\n",
    "    print(f\"   ↑ {row['feature']}: trend = {row['trend_coefficient']:.4f}\")\n",
    "\n",
    "print(\"\\n\" + \"=\"*60)\n",
    "print(\"✓ All results saved to reports/ directory\")\n",
    "print(\"=\"*60)"
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
with open('notebooks/08_final_pipeline.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("✓ Notebook created successfully at notebooks/08_final_pipeline.ipynb")
print("\n📋 This notebook requires the following modules to be implemented:")
print("  - src/data/preprocessing.py (basic_preprocessing_pipeline)")
print("  - src/features/feature_engineering.py (feature_engineering_pipeline)")
print("  - src/features/encoder.py (encode_categorical_columns)")
print("  - src/models/train_test_split.py (temporal split functions)")
print("  - src/models/smote_pipeline.py (apply_smote)")
print("  - src/models/boosting_models.py (train_xgboost, train_lightgbm, train_catboost)")
print("  - src/models/baseline_models.py (train_logistic_regression, train_random_forest, generate_predictions)")
print("  - src/evaluation/model_comparison.py (ModelComparison class)")
print("  - src/visualization/shap_visualizer.py (SHAP visualization functions)")
print("  - src/visualization/temporal_shap.py (temporal SHAP functions)")
print("  - src/visualization/save_figures.py (save_current_figure)")
print("\n⚠️  Make sure to install required packages:")
print("  pip install xgboost lightgbm catboost scikit-learn imbalanced-learn shap pandas matplotlib seaborn")
print("\n⚠️  Make sure to create these modules before running the notebook!")