"""Build a slide-level annotation table for the FN Brno prostate dataset.

Pipeline:
1. Parse each .czi slide filename ("<patient_ID>-24-<slide_No>-HE.czi") to get
   patient_id + slide_no.
2. Parse each annotation-table index string ("<patient_no>/<year> - <ranges>")
   into patient_no, year, and a list of (start, end) slide-number ranges.
   Ranges may be non-contiguous / have gaps to denote missing slides,
   e.g. "1-11,13-16" or "1,3-14".
3. Match each slide to its case record by (patient_id, slide_no) falling
   inside one of the case's ranges.
4. Derive slide-level and case-level labels (carcinoma, ISUP grade,
   Gleason score, perineural invasion) and assemble a flat output table.
"""

import re
import tempfile
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


# --------------------------------------------------------------------------- #
# Regex patterns
# --------------------------------------------------------------------------- #

SLIDE_NAME_PATTERN = re.compile(r"^(?P<patient_ID>\d+)-24-(?P<slide_No>\d+)-HE\.czi$")

# Annotation-table index, e.g. "68/24 - 1-11,13-16" or "86/24  1-10"
# (dash/spacing around the "-" before the ranges is inconsistent in the
# source table, hence the "\s*-?\s*" instead of a strict literal "-").
INDEX_PATTERN = re.compile(
    r"^\s*(?P<patient_no>\d+)\s*/\s*(?P<year>\d+)\s*-?\s*(?P<ranges>.+?)\s*$"
)


# --------------------------------------------------------------------------- #
# Range parsing / lookup helpers
# --------------------------------------------------------------------------- #


def parse_ranges(range_str: str) -> list[tuple[int, int]]:
    """Turn a comma-separated slide range string into a list of (start, end) tuples, inclusive on both ends."""
    ranges = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            ranges.append((int(start.strip()), int(end.strip())))
        else:
            n = int(part)
            ranges.append((n, n))
    return ranges


def parse_index(idx: str) -> dict[str, Any] | None:
    m = INDEX_PATTERN.match(idx)
    if not m:
        return None
    d = m.groupdict()
    return {
        "patient_no": int(d["patient_no"]),
        "year": int(d["year"]),
        "ranges": parse_ranges(d["ranges"]),
    }


def slide_in_ranges(slide_no: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= slide_no <= end for start, end in ranges)


def find_record(df: pd.DataFrame, patient_id: int, slide_no: int) -> pd.DataFrame:
    mask = (df["patient_no"] == patient_id) & df["ranges"].apply(
        lambda ranges: slide_in_ranges(slide_no, ranges)
    )
    return df[mask]


# --------------------------------------------------------------------------- #
# Slide-level parsing
# --------------------------------------------------------------------------- #


def parse_slide(
    slide_name: str, table: pd.DataFrame
) -> tuple[pd.Series, bool, bool] | None:
    match = SLIDE_NAME_PATTERN.match(slide_name)
    if not match:
        # Filename doesn't follow the expected naming convention.
        return None

    patient_id, slide_no = match.groups()
    record = find_record(table, int(patient_id), int(slide_no))

    if len(record) == 0:
        # Slide number isn't described in the table
        return None

    assert len(record) == 1, (
        f"Expected exactly one record for patient_id={patient_id}, "
        f"slide_no={slide_no}, got {len(record)}"
    )

    record = record.squeeze()

    case_is_cancer = (
        not pd.isna(record["Cancer_in_case"])
        and str(record["Cancer_in_case"]).lower() == "yes"
    )

    if not case_is_cancer:
        # Whole case is benign (or unlabeled) -> every slide is
        # automatically non-carcinoma.
        slide_label = False
    elif not pd.isna(record["Cancer_in_slides"]):
        # Case is malignant; carcinoma is present only on the listed slides.
        cancerous_slides = [
            s.strip() for s in str(record["Cancer_in_slides"]).split(",")
        ]
        slide_label = slide_no in cancerous_slides
    else:
        # Case is malignant but no specific slide list was given.
        slide_label = False

    return record, slide_label, case_is_cancer


# --------------------------------------------------------------------------- #
# Table assembly
# --------------------------------------------------------------------------- #


def create_df(slides: list[Path], annot_table: pd.DataFrame) -> pd.DataFrame:
    info: dict[str, list[Any]] = {
        "slide_path": [],
        "carcinoma": [],  # slide-level label
        "case_id": [],
        "patient_id": [],
        "isup_grade": [],
        "gleason_score": [],
        "case_carcinoma": [],  # case-level label
        "perineural_invasions": [],
    }

    for slide in slides:
        parsed = parse_slide(slide.name, annot_table)
        if parsed is None:
            continue

        record, slide_label, case_label = parsed

        info["slide_path"].append(str(slide))
        # Downstream scripts expect the slide-level annotation to be called "carcinoma"
        info["carcinoma"].append(slide_label)
        info["case_carcinoma"].append(case_label)
        info["case_id"].append(str(record["Biopsy_No"]).strip())
        info["patient_id"].append(record["patient_no"])

        pi = record["Perineural_invasion"]
        if pd.isna(pi):
            # Field is absent for negative cases -> treat as no invasion.
            info["perineural_invasions"].append(False)
        else:
            info["perineural_invasions"].append(
                slide_label and str(pi).lower() == "yes"
            )

        if slide_label and not pd.isna(record["Gleason_score"]):
            info["isup_grade"].append(record["ISUP_grade_ group"])
            info["gleason_score"].append(str(record["Gleason_score"]).strip())
        else:
            info["isup_grade"].append(None)
            info["gleason_score"].append(None)

    return pd.DataFrame(info)


@hydra.main(
    config_path="../configs", config_name="exploration/fn_brno", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = list(Path(config.data_dir).glob("*.czi"))

    annot_table = pd.read_excel(config.annot_table_path)
    # 1:1 mapping between cases and patients
    annot_table = annot_table.set_index("Patient_ID", drop=False)

    # Parse the "<patient_no>/<year> - <ranges>" index strings into
    # structured columns used by find_record().
    parsed = annot_table.index.map(parse_index)

    bad_rows = [
        idx for idx, p in zip(annot_table.index, parsed, strict=True) if p is None
    ]
    if bad_rows:
        raise ValueError(f"Could not parse annotation index for: {bad_rows}")

    annot_table["patient_no"] = [p["patient_no"] for p in parsed]
    annot_table["year"] = [p["year"] for p in parsed]
    annot_table["ranges"] = [p["ranges"] for p in parsed]

    explored = create_df(slides, annot_table)
    qc_exclude = pd.read_csv(config.qc_exclude_table_path)
    stems_to_exclude = set(qc_exclude["slide_stem"])
    explored = explored[
        ~(explored["slide_path"].map(lambda x: Path(x).stem).isin(stems_to_exclude))
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "fn_brno_prostate.csv"
        explored.to_csv(target, index=False)
        logger.log_artifact(str(target))


if __name__ == "__main__":
    main()
