"""
Main execution file for AccidentXAI.

Runs the complete research pipeline.
"""

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_pipeline():

    script_path = PROJECT_ROOT / "export_results.py"

    if not script_path.exists():

        raise FileNotFoundError(
            f"export_results.py not found at: {script_path}"
        )

    print("\nStarting AccidentXAI pipeline...\n")

    subprocess.run(
        ["python3", str(script_path)],
        check=True,
        cwd=PROJECT_ROOT
    )

    print(
        "\nPipeline execution completed.\n"
    )


if __name__ == "__main__":

    run_pipeline()