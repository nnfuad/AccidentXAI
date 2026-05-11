"""
Feature importance utilities for AccidentXAI.
"""

import pandas as pd
import numpy as np


def compute_mean_shap_importance(
    shap_values,
    X_sample
):
    """
    Compute mean absolute SHAP importance.
    """

    importance = np.abs(
        shap_values
    ).mean(axis=0)

    importance_df = pd.DataFrame({

        "Feature": X_sample.columns,

        "Mean_SHAP_Importance": importance

    })

    importance_df = importance_df.sort_values(
        by="Mean_SHAP_Importance",
        ascending=False
    )

    return importance_df