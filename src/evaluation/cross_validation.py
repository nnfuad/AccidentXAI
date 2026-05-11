"""
Cross-validation utilities for AccidentXAI.
"""

from sklearn.model_selection import cross_val_score


def run_cross_validation(
    model,
    X,
    y,
    scoring="f1_macro",
    cv=5
):
    """
    Run cross-validation.
    """

    print(f"\nRunning {cv}-Fold Cross Validation...\n")

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    print(f"Scores: {scores}")

    print(f"\nMean Score: {scores.mean():.4f}")

    print(f"Std Dev: {scores.std():.4f}")

    return scores