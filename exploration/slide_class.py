"""This file contains function for slide-level annotations (obtaining differs across datasets) and case identification."""

import re
from pathlib import Path

import pandas as pd


def get_case_id_mmci_tl(slide_path: Path) -> str:
    # File names are expected to be in the format: "PREFIX-YEAR_CASEID-SLIDENUMBER-[0,1].mrxs"
    return slide_path.stem.split("-")[1]


def carcinoma_bool_mmci_tl(slide_path: Path | str) -> bool:
    """Get the slide cancer status from the slide name from MMCI tile annotated data."""
    slide_path = Path(slide_path)

    if slide_path.stem[-1] not in ("0", "1"):
        raise ValueError(
            f"Invalid slide name: {slide_path.stem}. Expected format: *-[0,1].mrxs"
        )

    return slide_path.stem[-1] == "1"


def carcinoma_bool_mmci_sl(
    slide_path: Path | str, annotation_table: pd.DataFrame
) -> bool:
    """Get the slide carcinoma status from the slide name."""
    # <year>[_-]<bioptic_request>-<slide_number>.mrxs
    slide_name = Path(slide_path).name
    pattern = r"^(\d{4})[_-](\d+)-(\d+)\.mrxs$"
    match = re.match(pattern, slide_name)
    if not match:
        raise ValueError(f"Invalid filename format: {slide_name}")

    year, bioptic_request, slide_number = match.groups()
    r = bioptic_request.lstrip("0")
    index = f"{year[2:]}/{r}"
    record = annotation_table.loc[annotation_table["Biopsy_No"] == index]
    slide_number = slide_number.lstrip("0")

    if len(record) != 1:
        print("inconsistent", slide_name)
    record = record.squeeze()

    if record["Cancer_in_case"].lower() != "yes":
        return False

    if isinstance(record["Cancer_in_slides"], float):
        cancerous_slides = str(record["Cancer_in_slides"]).split(".")
    elif isinstance(record["Cancer_in_slides"], int):
        cancerous_slides = [str(record["Cancer_in_slides"])]
    else:
        cancerous_slides = record["Cancer_in_slides"].split(",")

    return slide_number in cancerous_slides
