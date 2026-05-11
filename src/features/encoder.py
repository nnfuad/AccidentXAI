"""
Encoding utilities for AccidentXAI.

This module handles:
- categorical encoding
- train/test consistency
- model-ready feature matrices
"""

import numpy as np

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

        # Convert safely to string
        X_train[col] = X_train[col].astype(str)

        X_test[col] = X_test[col].astype(str)

        # Fit ONLY on training data
        encoder.fit(X_train[col])

        # Transform training data
        X_train[col] = encoder.transform(
            X_train[col]
        )

        # Handle unseen labels in test set
        unseen_labels = np.setdiff1d(
            X_test[col].unique(),
            encoder.classes_
        )

        if len(unseen_labels) > 0:

            encoder.classes_ = np.concatenate(
                [encoder.classes_, unseen_labels]
            )

        # Transform test data
        X_test[col] = encoder.transform(
            X_test[col]
        )

        encoders[col] = encoder

    print("Encoding completed.\n")

    return X_train, X_test, encoders