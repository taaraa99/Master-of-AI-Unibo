import json
from pathlib import Path
from typing import Dict, Any


def save_result(
    sat_res_dir: Path,
    f_path: Path,
    approach: str,
    record: Dict[str, Any]
) -> None:
    """
    Merge the given record into the JSON file for the instance under sat_res_dir.

    Each instance file is named by its numeric index (digits extracted from f_path.stem), e.g., '3.json'.
    If the file already exists, load and update it; otherwise, create a new one.
    """
    # Extract numeric index from file stem, fallback to full stem
    digits = ''.join(filter(str.isdigit, f_path.stem))
    idx = int(digits) if digits else f_path.stem
    out_file = sat_res_dir / f"{idx}.json"

    # Load existing data if present
    if out_file.exists():
        try:
            with out_file.open('r', encoding='utf-8') as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # Insert or overwrite this approach's record
    data[approach] = record

    # Write back to disk with indentation
    with out_file.open('w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2)
