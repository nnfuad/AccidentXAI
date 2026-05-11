"""
Data loading utilities for AccidentXAI.
"""

from pathlib import Path

import pandas as pd
import yaml


def load_config():
    """
    Load YAML config file using absolute paths.
    """

    project_root = Path(__file__).resolve().parents[2]

    config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r") as file:

        config = yaml.safe_load(file)

    return config


def load_dataset(config):
    """
    Load dataset using absolute project paths.
    """

    project_root = Path(__file__).resolve().parents[2]

    relative_path = config["paths"]["raw_data"]

    data_path = project_root / relative_path

    if not data_path.exists():

        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    print(f"\nLoading dataset from:\n{data_path}\n")

    df = pd.read_csv(data_path)

    print("\nDataset loaded successfully.\n")

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

        print("Requested sample exceeds dataset size.")

        return df

    print(f"\nSampling {sample_size} rows...\n")

    sampled_df = df.sample(
        n=sample_size,
        random_state=config["project"]["random_state"]
    )

    return sampled_df


def dataset_overview(df):
    """
    Print dataset overview.
    """

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print(f"\nShape: {df.shape}")

    print("\nColumns:\n")
    print(df.columns.tolist())

    print("\nMissing Values:\n")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    print("\nData Types:\n")
    print(df.dtypes)

    print("\nSeverity Distribution:\n")

    if "Severity" in df.columns:

        print(df["Severity"].value_counts())

    print("\nOverview completed.\n")