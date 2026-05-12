"""
Model comparison utilities for AccidentXAI.

This module handles:
- metric aggregation
- model ranking
- comparison tables
- CSV export
"""

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


class ModelComparison:
    """
    Utility class for comparing ML models.
    """

    def __init__(self):

        self.results = []

    def add_model_result(
        self,
        model_name,
        y_true,
        y_pred,
        y_prob
    ):
        """
        Add model evaluation metrics.
        """

        result = {

            "Model": model_name,

            "F1_Macro": f1_score(
                y_true,
                y_pred,
                average="macro"
            ),

            "F1_Weighted": f1_score(
                y_true,
                y_pred,
                average="weighted"
            ),

            "Precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "ROC_AUC": roc_auc_score(
                y_true,
                y_prob
            )
        }

        self.results.append(result)

    def get_comparison_table(self):
        """
        Generate model comparison dataframe.
        """

        if len(self.results) == 0:

            raise ValueError(
                "No model results available."
            )

        comparison_df = pd.DataFrame(
            self.results
        )

        comparison_df = comparison_df.sort_values(
            by="ROC_AUC",
            ascending=False
        ).reset_index(drop=True)

        return comparison_df

    def save_results(
        self,
        save_path="reports/tables/model_comparison.csv"
    ):
        """
        Save comparison table safely.
        """

        comparison_df = self.get_comparison_table()

        save_path = Path(save_path)

        # Create directories safely
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        comparison_df.to_csv(
            save_path,
            index=False
        )

        print(
            f"\nModel comparison saved to:\n{save_path}\n"
        )

        return comparison_df