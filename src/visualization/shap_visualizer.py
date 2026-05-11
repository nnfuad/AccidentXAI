"""
SHAP visualization utilities for AccidentXAI.

This module handles:
- SHAP explainer creation
- summary plots
- dependence plots
- waterfall plots
"""

import shap
import matplotlib.pyplot as plt


def create_tree_explainer(model):
    """
    Create SHAP TreeExplainer.
    """

    print("\nCreating SHAP explainer...\n")

    explainer = shap.TreeExplainer(model)

    return explainer


def compute_shap_values(
    explainer,
    X_sample
):
    """
    Compute SHAP values.
    """

    print("\nComputing SHAP values...\n")

    shap_values = explainer.shap_values(X_sample)

    print("SHAP computation completed.\n")

    return shap_values


def plot_shap_summary(
    shap_values,
    X_sample
):
    """
    Plot SHAP summary plot.
    """

    print("\nGenerating SHAP summary plot...\n")

    shap.summary_plot(
        shap_values,
        X_sample
    )


def plot_shap_dependence(
    shap_values,
    X_sample,
    feature_name
):
    """
    Plot SHAP dependence plot.
    """

    print(f"\nGenerating dependence plot for {feature_name}...\n")

    shap.dependence_plot(
        feature_name,
        shap_values,
        X_sample
    )


def plot_shap_waterfall(
    explainer,
    shap_values,
    X_sample,
    sample_index=0
):
    """
    Plot SHAP waterfall plot for local explanation.
    """

    print("\nGenerating SHAP waterfall plot...\n")

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[sample_index],
        X_sample.iloc[sample_index]
    )

    plt.show()