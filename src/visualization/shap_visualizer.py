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
    Compute SHAP values using modern SHAP API.
    """

    print("\nComputing SHAP values...\n")

    # Modern SHAP API
    shap_explanation = explainer(
        X_sample
    )

    shap_values = shap_explanation.values

    # Binary classification compatibility
    if len(shap_values.shape) == 3:

        shap_values = shap_values[:, :, 1]

    print("SHAP computation completed.\n")

    return shap_explanation, shap_values


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

    print(
        f"\nGenerating dependence plot for "
        f"{feature_name}...\n"
    )

    shap.dependence_plot(
        feature_name,
        shap_values,
        X_sample
    )


def plot_shap_waterfall(
    shap_explanation,
    sample_index=0
):
    """
    Plot local SHAP waterfall explanation.
    """

    print(
        "\nGenerating SHAP waterfall plot...\n"
    )

    shap.plots.waterfall(
        shap_explanation[sample_index]
    )

    plt.show()