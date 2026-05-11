"""
Simple logging utility for AccidentXAI.
"""

from datetime import datetime


def log(message):
    """
    Print timestamped logs.
    """

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{current_time}] {message}")