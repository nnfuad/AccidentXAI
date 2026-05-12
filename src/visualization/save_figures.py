"""
Figure saving utilities for AccidentXAI.

This module standardizes:
- figure saving
- DPI settings
- output organization
"""

from pathlib import Path

import matplotlib.pyplot as plt


FIGURE_DIR = Path("reports/figures")

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def save_current_figure(
    filename,
    dpi=300
):
    """
    Save currently active matplotlib figure.
    """

    save_path = FIGURE_DIR / filename

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    print(
        f"\nFigure saved to:\n{save_path}\n"
    )


def save_and_show(
    filename,
    dpi=300
):
    """
    Save and display figure.
    """

    save_current_figure(
        filename,
        dpi
    )

    plt.show()