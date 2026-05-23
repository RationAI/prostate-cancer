# credits: https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/lymph-nodes/-/blob/develop/preprocessing/qc.py?ref_type=heads

import asyncio
import shutil
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
    store_masks_at_original_resolution: bool


async def qc_main(
    output_path: Path,
    slides: list[str],
    logger: MLFlowLogger,
    request_timeout: int,
    max_concurrent: int,
    qc_parameters: QCParameters,
) -> None:
    async with rationai.AsyncClient(
        qc_base_url="http://rayservice-qc-update-trial-serve-svc.rationai-jobs-ns.svc.cluster.local:8000/"
    ) as client:  # type: ignore[attr-defined]
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

        # Merge generated csv files
        csvs = list(Path(output_path).glob("*.csv"))

        if len(csvs) > 0:
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
        )
    )


if __name__ == "__main__":
    main()
