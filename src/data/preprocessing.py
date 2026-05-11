"""
Preprocessing utilities for AccidentXAI.

This module handles:
- missing values
- column filtering
- datetime conversion
- duplicate removal
- basic cleaning
"""

import pandas as pd


def convert_datetime(df, datetime_column="Start_Time"):
    """
    Convert datetime column to pandas datetime format.
    """

    print(f"Converting {datetime_column} to datetime format...")

    df[datetime_column] = pd.to_datetime(
        df[datetime_column],
        errors="coerce"
    )

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """

    initial_shape = df.shape[0]

    df = df.drop_duplicates()

    final_shape = df.shape[0]

    print(f"Removed {initial_shape - final_shape} duplicate rows.")

    return df


def drop_high_missing_columns(df, threshold=0.7):
    """
    Drop columns with excessive missing values.

    threshold=0.7 means:
    drop columns with >70% missing values.
    """

    missing_ratio = df.isnull().mean()

    columns_to_drop = missing_ratio[
        missing_ratio > threshold
    ].index.tolist()

    print("\nDropping high-missing columns:\n")
    print(columns_to_drop)

    df = df.drop(columns=columns_to_drop)

    return df


def fill_missing_values(df):
    """
    Fill missing values using simple strategies.

    Numerical columns:
    -> median

    Categorical columns:
    -> mode
    """

    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_cols = df.select_dtypes(
        include=["object", "bool"]
    ).columns

    # Numerical columns
    for col in numerical_cols:

        median_value = df[col].median()

        df[col] = df[col].fillna(median_value)

    # Categorical columns
    for col in categorical_cols:

        mode_value = df[col].mode()[0]

        df[col] = df[col].fillna(mode_value)

    print("\nMissing values handled successfully.")

    return df


def select_relevant_columns(df):
    """
    Select columns useful for severity prediction.

    This prevents:
    - leakage
    - unnecessary memory usage
    - irrelevant features
    """

    selected_columns = [

        # Target
        "Severity",

        # Time
        "Start_Time",

        # Weather
        "Temperature(F)",
        "Humidity(%)",
        "Pressure(in)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Weather_Condition",

        # Road / Environment
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",

        # Civil Twilight
        "Sunrise_Sunset",

        # Location
        "State",

        # Coordinates
        "Start_Lat",
        "Start_Lng"
    ]

    existing_columns = [
        col for col in selected_columns
        if col in df.columns
    ]

    df = df[existing_columns]

    print("\nSelected relevant columns.")

    return df


def basic_preprocessing_pipeline(df):
    """
    Execute basic preprocessing pipeline.
    """

    print("\nStarting preprocessing pipeline...\n")

    df = convert_datetime(df)

    df = remove_duplicates(df)

    df = select_relevant_columns(df)

    df = drop_high_missing_columns(df)

    df = fill_missing_values(df)

    print("\nPreprocessing completed successfully.\n")

    return df