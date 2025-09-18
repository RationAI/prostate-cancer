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
        self.all_preds: list[NDArray[np.float32]] = []
        self.all_labels: list[NDArray[np.float32]] = []

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

        positive_counts = [0] * 21
        negative_counts = [0] * 21

        for score, label in zip(preds, labels, strict=False):
            bin_index = int(score * 20)
            (positive_counts if label == 1 else negative_counts)[bin_index] += 1

        bar_width = 0.4
        x = list(range(21))

        _, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x, positive_counts, bar_width, label="Positive")
        ax.bar([i + bar_width for i in x], negative_counts, bar_width, label="Negative")

        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Count")
        ax.set_title("Tile Predictions Histogram")
        ax.set_xticks([i + bar_width / 2 for i in range(0, 21, 2)])
        ax.set_xticklabels([f"{i / 20:.2f}" for i in range(0, 21, 2)])

        ax.legend()

        plot_path = Path("histogram.png")
        plt.savefig(plot_path)

        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        plot_path.unlink(missing_ok=True)

        self.all_preds.clear()
        self.all_labels.clear()
