# credits: https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes/-/blob/develop/preprocessing/qc.py?ref_type=heads

import asyncio
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import hydra
import pandas as pd
import rationai
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.types import SlideCheckConfig
from tqdm.asyncio import tqdm


class QCParameters(TypedDict):
    check_residual: bool
    check_folding: bool
    check_blur: bool
    wb_correction: bool
    subsample_masks: bool


def get_qc_masks(qc_parameters: QCParameters) -> Generator[tuple[str, str], None, None]:
    if qc_parameters["check_blur"]:
        yield ("Piqe/blur_score_coverage", "blur_per_tile")
        yield ("Piqe/blur_score_per_pixel", "blur_per_pixel")

    if qc_parameters["check_residual"]:
        yield ("ResidualArtifactsAndCoverage/artifacts_coverage", "residual_per_tile")
        yield ("ResidualArtifactsAndCoverage/artifacts_per_pixel", "residual_per_pixel")

    if qc_parameters["check_folding"]:
        yield ("FoldingFunction/folding_per_pixel", "folding_per_pixel")


def organize_masks(output_path: Path, subdir: str, current_subdir: str) -> None:
    prefix_dir = output_path / subdir
    prefix_dir.mkdir(parents=True, exist_ok=True)

    current_dir = output_path / current_subdir

    for file in list(current_dir.glob("*.tiff")):
        destination = prefix_dir / file.name
        file.rename(destination)


async def qc_main(
    output_path: Path,
    slides: list[str],
    logger: MLFlowLogger,
    request_timeout: int,
    max_concurrent: int,
    qc_parameters: QCParameters,
    base_url: str,
) -> None:
    async with rationai.AsyncClient(qc_base_url=base_url) as client:  # type: ignore[attr-defined]
        async for result in tqdm(
            client.qc.check_slides(
                slides,
                output_path,
                config=SlideCheckConfig(**qc_parameters),
                timeout=request_timeout,
                max_concurrent=max_concurrent,
            ),
            total=len(slides),
        ):
            if not result.success:
                with open(output_path / "qc_errors.log", "a") as log_file:
                    log_file.write(
                        f"Failed to process {result.wsi_path}: {result.error}\n"
                    )

        # Organize generated masks into subdirectories
        for prefix, artifact_name in get_qc_masks(qc_parameters):
            organize_masks(Path(output_path), artifact_name, prefix)

        # Merge generated csv files
        csvs = list(Path(output_path).rglob("*.csv"))
        if len(csvs) > 1:  # if only one, keep it
            pd.concat([pd.read_csv(f) for f in csvs]).to_csv(
                Path(output_path, "qc_metrics.csv"), index=False
            )

            # Remove individual csv files
            for f in csvs:
                f.unlink()

        logger.log_artifacts(local_dir=str(output_path))


@with_cli_args(["+preprocessing=quality_control_v2"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset = pd.read_csv(download_artifacts(config.data.metadata_table))

    output_path = Path(config.output_path)
    if output_path.exists():
        shutil.rmtree(str(output_path))

    output_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        qc_main(
            output_path=output_path,
            slides=dataset["slide_path"].to_list(),
            logger=logger,
            request_timeout=config.request_timeout,
            max_concurrent=config.max_concurrent,
            qc_parameters=config.qc_parameters,
            base_url=config.base_url,
        )
    )


if __name__ == "__main__":
    main()
