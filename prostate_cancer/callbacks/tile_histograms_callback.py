from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from lightning import Callback, LightningModule, Trainer
from numpy.typing import NDArray
from rationai.mlkit.lightning.loggers import MLFlowLogger

from prostate_cancer.typing import LabeledSampleBatch


class TileHistogramsCallback(Callback):
    def __init__(self) -> None:
        """This callback creates prediction histograms for both negative and positive distribution of tiles."""
        super().__init__()
        self.all_preds: list[NDArray[np.floating]] = []
        self.all_labels: list[NDArray[np.floating]] = []

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: LabeledSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, y, _ = batch
        preds = outputs.detach().cpu().numpy().flatten()
        labels = y.detach().cpu().numpy().flatten()

        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert isinstance(trainer.logger, MLFlowLogger)

        preds = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)

        pos_preds = preds[labels == 1]
        neg_preds = preds[labels == 0]

        # Create 2 histograms with independent y-axes
        _, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(14, 6))

        ax_pos.hist(pos_preds, bins=20, range=(0, 1), color="green", alpha=0.7)
        ax_pos.set_title("Positive Samples")
        ax_pos.set_xlabel("Predicted Score")
        ax_pos.set_ylabel("Count")

        ax_neg.hist(neg_preds, bins=20, range=(0, 1), color="red", alpha=0.7)
        ax_neg.set_title("Negative Samples")
        ax_neg.set_xlabel("Predicted Score")
        ax_neg.set_ylabel("Count")

        plt.suptitle("Predicted Score Histograms by Class")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plot_path = Path("histograms.png")
        plt.savefig(plot_path)

        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        plot_path.unlink(missing_ok=True)
        plt.close()

        self.all_preds.clear()
        self.all_labels.clear()
