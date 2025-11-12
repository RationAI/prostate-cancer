"""This file contains function for slide-level annotations (obtaining differs across datasets) and case identification. It should be used only in the exploration phase."""

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


def parse_slide_name(slide_name: str) -> tuple[str, str, str] | None:
    # <year>[_-]<bioptic_request>-<slide_number>.mrxs
    pattern = r"^(\d{4})[_-](\d+)-(\d+)\.mrxs$"
    match = re.match(pattern, slide_name)
    if not match:
        return None

    year, bioptic_request, slide_number = match.groups()
    return year, bioptic_request, slide_number


def get_record_by_slide(table: pd.DataFrame, slide_name: str) -> pd.Series | None:
    parsed = parse_slide_name(slide_name)
    if parsed is None:
        return None

    year, bioptic_request, _ = parsed
    r = bioptic_request.lstrip("0")
    index = f"{year[2:]}/{r}"
    record = table.loc[table["Biopsy_No"] == index]
    record = record.squeeze()
    return record


def carcinoma_bool_mmci_sl(
    slide_path: Path | str, annotation_table: pd.DataFrame
) -> bool | None:
    """Get the slide carcinoma status from the slide name."""
    slide_name = Path(slide_path).name

    parsed = parse_slide_name(slide_name)
    if parsed is None:
        return None

    _, _, slide_number = parsed
    record = get_record_by_slide(annotation_table, slide_name)
    if record is None:
        return None

    if record["Cancer_in_case"].lower() != "yes":
        return False

    # handling various typos and edge cases in the table
    if isinstance(record["Cancer_in_slides"], float):
        cancerous_slides = str(record["Cancer_in_slides"]).split(".")
    elif isinstance(record["Cancer_in_slides"], int):
        cancerous_slides = [str(record["Cancer_in_slides"])]
    else:
        cancerous_slides = record["Cancer_in_slides"].split(",")

    return slide_number in cancerous_slides
