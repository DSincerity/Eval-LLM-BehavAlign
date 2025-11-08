"""File I/O utilities."""
import json
import time
from typing import Any, Dict


def load_txt_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Load a text file and return its content.

    Args:
        file_path: Path to text file
        encoding: Text encoding (default: utf-8)

    Returns:
        File content as string
    """
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def load_json(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """Load a JSON file to a dictionary.

    Args:
        file_path: Path to JSON file
        encoding: Text encoding (default: utf-8)

    Returns:
        Parsed JSON as dictionary
    """
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    print(f"Data has been successfully loaded from {file_path}")
    return data


def save_dict_to_json(data: Dict[str, Any], file_path: str, encoding: str = 'utf-8') -> None:
    """Save a dictionary to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to output JSON file
        encoding: Text encoding (default: utf-8)
    """
    try:
        with open(file_path, 'w', encoding=encoding) as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data has been successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def compute_time(start_time: float) -> float:
    """Compute elapsed time in minutes since start_time.

    Args:
        start_time: Start time from time.time()

    Returns:
        Elapsed time in minutes
    """
    return (time.time() - start_time) / 60.0
