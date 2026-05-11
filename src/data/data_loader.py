"""
Data loading utilities for AccidentXAI.

This module handles:
- reading datasets
- optional sampling
- basic dataset inspection
"""

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path="configs/config.yaml"):
    """
    Load YAML configuration file.
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def load_dataset(config):
    """
    Load accident dataset using config paths.
    """

    data_path = Path(config["paths"]["raw_data"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    print(f"\nLoading dataset from:\n{data_path}\n")

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully.\n")

    return df


def sample_dataset(df, config):
    """
    Sample dataset for manageable experimentation.
    """

    sampling_enabled = config["sampling"]["enabled"]

    if not sampling_enabled:
        return df

    sample_size = config["sampling"]["sample_size"]

    if sample_size >= len(df):
        print("Requested sample size exceeds dataset size.")
        return df

    print(f"Sampling {sample_size} rows...\n")

    sampled_df = df.sample(
        n=sample_size,
        random_state=config["project"]["random_state"]
    )

    return sampled_df


def dataset_overview(df):
    """
    Print basic dataset information.
    """

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"\nShape: {df.shape}")

    print("\nColumns:\n")
    print(df.columns.tolist())

    print("\nMissing Values:\n")
    print(df.isnull().sum().sort_values(ascending=False).head(15))

    print("\nData Types:\n")
    print(df.dtypes)

    print("\nClass Distribution:\n")

    if "Severity" in df.columns:
        print(df["Severity"].value_counts())

    print("\nOverview completed.\n")