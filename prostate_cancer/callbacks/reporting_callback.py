# Third-party imports
import lightning
import mlflow
from lightning import Callback

# Local Imports
from report.masks import AbstractMasksRetriever, RunIDMlflowMaskRetriever
from report.reporter import ReportAssembler
from report.save import RunIDMLFlowReportAttacher


class Reporter(Callback):
    """A callback that assembles an HTML presentation of the test results."""

    def __init__(
        self,
        background: AbstractMasksRetriever,
        masks: list[AbstractMasksRetriever],
        save_dir: str,
        title: str,
        template_asset_path: str,
    ):
        super().__init__()
        self.background_retriever = background
        self.mask_retrievers = masks
        self.save_dir = save_dir
        self.title = title
        self.template_asset_path = template_asset_path

    def report(self):
        run_id = mlflow.active_run().info.run_id
        storer = RunIDMLFlowReportAttacher(artifact_path=self.save_dir, run_id=run_id)
        for mask_retriever in self.mask_retrievers:
            if hasattr(mask_retriever, "run_id") and mask_retriever.run_id is None:
                mask_retriever.run_id = run_id
        report_assembler = ReportAssembler(
            title=self.title,
            background=self.background_retriever,
            save=storer,
            metrics_run_ids=[run_id],
            mask_retrievers=self.mask_retrievers,
            template_asset_path=self.template_asset_path,
        )
        report_assembler.assemble_report()

    def on_test_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self.report()

    def on_fit_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self.report()

    def on_train_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self.report()
