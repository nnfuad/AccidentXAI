"""
Temporal SHAP analysis utilities for AccidentXAI.

This module handles:
- yearly SHAP analysis
- temporal feature importance tracking
- SHAP drift visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import shap


def compute_yearly_shap_importance(

    model,

    df,

    feature_columns,

    encoders,

    year_column="Year",

    sample_size=500

):

    """

    Compute yearly SHAP importance safely.

    Applies SAME encoding used during training.

    """

    print("\nComputing yearly SHAP importance...\n")

    explainer = shap.TreeExplainer(model)

    yearly_results = []

    unique_years = sorted(

        df[year_column].unique()

    )

    for year in unique_years:

        print(f"Processing Year: {year}")

        yearly_df = df[

            df[year_column] == year

        ]

        if len(yearly_df) < sample_size:

            sample_df = yearly_df.copy()

        else:

            sample_df = yearly_df.sample(

                n=sample_size,

                random_state=42

            )

        # Select feature columns only

        X_year = sample_df[

            feature_columns

        ].copy()

        # Apply encoders

        for col, encoder in encoders.items():

            X_year[col] = X_year[col].astype(str)

            unseen_labels = set(

                X_year[col]

            ) - set(encoder.classes_)

            if unseen_labels:

                import numpy as np

                encoder.classes_ = np.concatenate(

                    [

                        encoder.classes_,

                        list(unseen_labels)

                    ]

                )

            X_year[col] = encoder.transform(

                X_year[col]

            )

        # Compute SHAP values

        shap_values = explainer.shap_values(

            X_year

        )

        mean_importance = np.abs(

            shap_values

        ).mean(axis=0)

        temp_df = pd.DataFrame({

            "Feature": feature_columns,

            "Importance": mean_importance,

            "Year": year

        })

        yearly_results.append(temp_df)

    final_df = pd.concat(

        yearly_results,

        ignore_index=True

    )

    return final_df


def plot_temporal_feature_drift(
    yearly_shap_df,
    top_n=5
):
    """
    Plot temporal SHAP drift.
    """

    feature_ranking = (
        yearly_shap_df
        .groupby("Feature")["Importance"]
        .mean()
        .sort_values(ascending=False)
    )

    top_features = feature_ranking.head(top_n).index

    filtered_df = yearly_shap_df[
        yearly_shap_df["Feature"].isin(top_features)
    ]

    plt.figure(figsize=(12, 6))

    for feature in top_features:

        temp = filtered_df[
            filtered_df["Feature"] == feature
        ]

        plt.plot(
            temp["Year"],
            temp["Importance"],
            marker="o",
            label=feature
        )

    plt.title(
        "Temporal SHAP Feature Drift"
    )

    plt.xlabel("Year")

    plt.ylabel("Mean SHAP Importance")

    plt.legend()

    plt.grid(True)

    plt.show()


def get_top_features_by_year(
    yearly_shap_df,
    top_n=10
):
    """
    Get top SHAP features for each year.
    """

    results = {}

    years = yearly_shap_df["Year"].unique()

    for year in years:

        temp = yearly_shap_df[
            yearly_shap_df["Year"] == year
        ]

        top_features = temp.sort_values(
            by="Importance",
            ascending=False
        ).head(top_n)

        results[year] = top_features

    return results