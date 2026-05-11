"""
Encoding utilities for AccidentXAI.

This module handles:
- categorical encoding
- train/test consistency
- model-ready feature matrices
"""

from sklearn.preprocessing import LabelEncoder


def encode_categorical_columns(
    X_train,
    X_test
):
    """
    Encode categorical columns using Label Encoding.

    IMPORTANT:
    Fit encoder ONLY on training data
    to avoid leakage.
    """

    print("\nEncoding categorical columns...\n")

    categorical_columns = X_train.select_dtypes(
        include=["object"]
    ).columns

    encoders = {}

    for col in categorical_columns:

        encoder = LabelEncoder()

        # Convert to string for safety
        X_train[col] = X_train[col].astype(str)

        X_test[col] = X_test[col].astype(str)

        # Fit ONLY on train
        encoder.fit(X_train[col])

        # Transform train
        X_train[col] = encoder.transform(
            X_train[col]
        )

        # Handle unseen labels safely
        unseen_labels = set(X_test[col]) - set(encoder.classes_)

        if unseen_labels:

            encoder.classes_ = list(encoder.classes_) + list(unseen_labels)

        # Transform test
        X_test[col] = encoder.transform(
            X_test[col]
        )

        encoders[col] = encoder

    print("Encoding completed.\n")

    return X_train, X_test, encoders