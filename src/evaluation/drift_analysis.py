"""
Drift analysis utilities for AccidentXAI.
"""

import pandas as pd


def compute_feature_drift(
    yearly_shap_df
):
    """
    Compute feature importance drift range.
    """

    drift_df = (
        yearly_shap_df
        .groupby("Feature")["Importance"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )

    drift_df["Drift_Range"] = (
        drift_df["max"] - drift_df["min"]
    )

    drift_df = drift_df.sort_values(
        by="Drift_Range",
        ascending=False
    )

    return drift_df