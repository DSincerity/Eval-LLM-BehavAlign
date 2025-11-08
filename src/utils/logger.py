"""Logging utilities for simulation runs."""
import sys


class Logger:
    """Logger class to log messages to a file and optionally to the console."""

    def __init__(self, log_file: str, verbose: bool = True):
        """Initialize logger.

        Args:
            log_file: Path to log file
            verbose: Whether to print to console in addition to file
        """
        self.terminal = sys.stdout
        self.log = open(log_file, "w")
        self.verbose = verbose

        self.write(f"All outputs written to {log_file}")

    def write(self, message: str):
        """Write message to log file and optionally to console.

        Args:
            message: Message to write
        """
        self.log.write(message + '\n')
        if self.verbose:
            self.terminal.write(message + '\n')

    def flush(self):
        """Flush buffers (required for file-like interface)."""
        pass

    def __del__(self):
        """Close log file on deletion."""
        if hasattr(self, 'log'):
            self.log.close()
