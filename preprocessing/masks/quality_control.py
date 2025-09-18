import asyncio
import tempfile
from pathlib import Path
from typing import Any, cast

import hydra
from aiohttp import ClientSession, ClientTimeout
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit.autolog import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


async def put_request(
    session: ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    request_timeout: int,
    data: dict[str, Any],
) -> tuple[int, str]:
    timeout = ClientTimeout(total=request_timeout)

    try:
        async with semaphore, session.put(url, json=data, timeout=timeout) as response:
            result = await response.text()

            print(
                f"Processed {data['wsi_path']}:\n\tStatus: {response.status} \n\tResponse: {result}\n"
            )

            return response.status, result
    except TimeoutError:
        slide_name = Path(data["wsi_path"]).name
        print(
            f"Request to {url} timed out after {request_timeout} seconds. Slide: {slide_name}"
        )
        return -1, "Timeout"


async def repeatable_put_request(
    session: ClientSession,
    url: str,
    data: dict[str, Any],
    num_repeats: int,
    semaphore: asyncio.Semaphore,
    request_timeout: int,
) -> None:
    for attempt in range(1, num_repeats + 1):
        status, text = await put_request(session, url, semaphore, request_timeout, data)

        if status == -1 and text == "Timeout":
            return

        if status == 500 and text == "Internal Server Error":
            att_count = f"attempt {attempt}/{num_repeats}"
            print(
                f"Unexpected status 500 received for {data['wsi_path']} ({att_count}):\n\tResponse: {text}\n"
            )
            await asyncio.sleep(2**attempt)

            continue

        print(
            f"Processed {data['wsi_path']}:\n\tStatus: {status} \n\tResponse: {text}\n"
        )

        return

    print(f"Failed to process {data['wsi_path']}:\n\tAll retry attempts failed\n")


async def generate_report(
    session: ClientSession,
    report_request_timeout: int,
    slides: list[Path],
    output_dir: str,
    save_location: str,
    url: str,
    semaphore: asyncio.Semaphore,
) -> None:
    url = url + "report"

    data = {
        "backgrounds": [str(slide) for slide in slides],
        "mask_dir": output_dir,
        "save_location": save_location,
    }

    try:
        async with (
            semaphore,
            session.put(
                url, json=data, timeout=ClientTimeout(total=report_request_timeout)
            ) as response,
        ):
            result = await response.text()

            print(
                f"Report generation:\n\tStatus: {response.status} \n\tResponse: {result}\n"
            )
    except TimeoutError:
        print(
            f"Report generation request to {url} timed out after {report_request_timeout} seconds."
        )


async def qc_main(
    output_path: str,
    report_path: str,
    slides: list[Path],
    mask_level: int,
    sample_level: int,
    logger: MLFlowLogger,
    url: str,
    semaphore: asyncio.Semaphore,
    request_timeout: int,
    report_request_timeout: int,
    num_repeats: int,
) -> None:
    async with ClientSession() as session:
        tasks = [
            repeatable_put_request(
                session=session,
                request_timeout=request_timeout,
                url=url,
                num_repeats=num_repeats,
                semaphore=semaphore,
                data={
                    "wsi_path": str(slide),
                    "output_path": output_path,
                    "mask_level": mask_level,
                    "sample_level": sample_level,
                    "check_residual": True,
                    "check_folding": True,
                    "check_focus": True,
                },
            )
            for slide in slides
        ]

        await asyncio.gather(*tasks)

        await generate_report(
            session=session,
            slides=slides,
            output_dir=output_path,
            save_location=report_path,
            url=url,
            semaphore=semaphore,
            report_request_timeout=report_request_timeout,
        )

        logger.experiment.log_artifacts(run_id=logger.run_id, local_dir=report_path)


@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)

    output_path = Path(config.qc_masks.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    prostate_cancer_path = config.prostate_cancer_path

    slides = list(Path(config.data_path).resolve().rglob("*.mrxs")) + list(
        Path(config.test_data_path).resolve().rglob("*.mrxs")
    )

    semaphore = asyncio.Semaphore(config.qc_masks.request_limit)

    with tempfile.TemporaryDirectory(
        prefix="qc_masks_report_", dir=Path(prostate_cancer_path).as_posix()
    ) as tmp_dir:  # Create a temporary directory for the report
        report_path = Path(tmp_dir) / "report.html"

        output_path.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        asyncio.run(
            qc_main(
                output_path=output_path.absolute().as_posix(),
                report_path=report_path.absolute().as_posix(),
                slides=slides,
                logger=logger,
                url=config.qc_masks.url,
                mask_level=config.qc_masks.mask_level,
                sample_level=config.qc_masks.sample_level,
                semaphore=semaphore,
                request_timeout=config.qc_masks.request_timeout,
                report_request_timeout=config.qc_masks.report_request_timeout,
                num_repeats=config.qc_masks.num_repeats,
            )
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
