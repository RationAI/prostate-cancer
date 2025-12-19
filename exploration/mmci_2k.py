import re
import tempfile
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


# avoid repetitive regex build
# <year>[_-]<bioptic_request>-<slide_number>.mrxs
PATTERN = re.compile(r"^(\d{4})[_-](\d+)-(\d+)\.mrxs$")


def parse_slide(slide_name: str, table: pd.DataFrame) -> tuple[pd.Series, bool] | None:
    match = PATTERN.match(slide_name)
    if not match:
        return None

    year, bioptic_request, slide_number = match.groups()
    r = bioptic_request.lstrip("0")
    index = f"{year[2:]}/{r}"
    record = table.loc[index]
    record = record.squeeze()

    if record["Cancer_in_case"].lower() != "yes":
        slide_label = False
    else:
        # handling various typos and edge cases in the table
        if isinstance(record["Cancer_in_slides"], float):
            cancerous_slides = str(record["Cancer_in_slides"]).split(".")
        elif isinstance(record["Cancer_in_slides"], int):
            cancerous_slides = [str(record["Cancer_in_slides"])]
        else:
            cancerous_slides = record["Cancer_in_slides"].split(",")

        slide_label = slide_number in cancerous_slides

    return record, slide_label


def create_df(slides: list[Path], annot_table: pd.DataFrame) -> pd.DataFrame:
    info: dict[str, list[Any]] = {
        "slide_path": [],
        "carcinoma": [],
        "case_id": [],
        "patient_id": [],
        "isup_grade": [],
        "gleason_score": [],
        "case_carcinoma": [],
        "perineural_invasions": [],
    }

    for slide in slides:
        slide_name = slide.name
        parsed = parse_slide(slide_name, annot_table)
        if parsed is None:
            continue

        record, slide_label = parsed
        case_label = record["Cancer_in_case"].lower() == "yes"

        info["slide_path"].append(str(slide))

        # some of the further scripts expect slide annotation to be called carcinoma
        info["carcinoma"].append(slide_label)
        info["case_carcinoma"].append(case_label)
        info["case_id"].append(record["Biopsy_No"].strip())
        info["patient_id"].append(record["Patient_ID"].strip())

        pi = record["Perineural_invasion"]

        # for negative cases its not present
        if pd.isna(pi):
            info["perineural_invasions"].append(False)
        else:
            info["perineural_invasions"].append(slide_label and pi.lower() == "yes")

        if slide_label and not pd.isna(record["Gleason_score"]):
            info["isup_grade"].append(record["ISUP_grade_ group"])
            info["gleason_score"].append(record["Gleason_score"].strip())
        else:
            info["isup_grade"].append(None)
            info["gleason_score"].append(None)

    return pd.DataFrame(info)


@hydra.main(
    config_path="../configs", config_name="exploration/mmci_2k", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = Path(config.data_dir).glob("*.mrxs")
    annot_table = pd.read_excel(config.annot_table_path)
    annot_table = annot_table.set_index("Biopsy_No", drop=False)
    explored = create_df(list(slides), annot_table)

    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "mmci_2k_prostate.csv"
        explored.to_csv(target, index=False)
        logger.log_artifact(str(target))


if __name__ == "__main__":
    main()
