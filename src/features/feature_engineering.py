"""
Feature engineering utilities for AccidentXAI.

This module creates:
- temporal features
- encoded target variables
- additional model-ready features
"""

import pandas as pd


def create_time_features(df, datetime_column="Start_Time"):
    """
    Create useful temporal features.
    """

    print("Creating temporal features...\n")

    df["Year"] = df[datetime_column].dt.year

    df["Month"] = df[datetime_column].dt.month

    df["Day"] = df[datetime_column].dt.day

    df["Hour"] = df[datetime_column].dt.hour

    df["Weekday"] = df[datetime_column].dt.weekday

    return df


def create_rush_hour_feature(df):
    """
    Create rush-hour indicator feature.
    """

    print("Creating rush-hour feature...\n")

    df["Rush_Hour"] = df["Hour"].apply(
        lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0
    )

    return df


def create_night_feature(df):
    """
    Create night-driving indicator.
    """

    print("Creating night-driving feature...\n")

    df["Night_Driving"] = df["Hour"].apply(
        lambda x: 1 if (x >= 20 or x <= 5) else 0
    )

    return df


def simplify_target(df):
    """
    Convert Severity into binary classification.

    Original severity:
    1, 2 = Less severe
    3, 4 = Severe

    Binary target:
    0 = Non-Severe
    1 = Severe
    """

    print("Simplifying target variable...\n")

    df["Severity_Binary"] = df["Severity"].apply(
        lambda x: 1 if x >= 3 else 0
    )

    return df


def drop_unused_columns(df):
    """
    Drop columns no longer needed after feature engineering.
    """

    columns_to_drop = [
        "Start_Time",
        "Severity"
    ]

    existing_columns = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    df = df.drop(columns=existing_columns)

    return df


def feature_engineering_pipeline(df):
    """
    Full feature engineering pipeline.
    """

    print("\nStarting feature engineering pipeline...\n")

    df = create_time_features(df)

    df = create_rush_hour_feature(df)

    df = create_night_feature(df)

    df = simplify_target(df)

    df = drop_unused_columns(df)

    print("\nFeature engineering completed.\n")

    return df