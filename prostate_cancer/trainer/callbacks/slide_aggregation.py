# Copyright (c) The RationAI team.

import functools
import logging

import lightning
import mlflow
import numpy as np
import pandas as pd
import pyvips
from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve

from prostate_cancer.trainer.callbacks.heatmap_visualizer import HeatmapVisualizer
from prostate_cancer.trainer.callbacks.image_builders import InMemoryHeatmapAssembler
from prostate_cancer.trainer.callbacks.vis_mode import IdentityMode


logger = logging.getLogger("callbacks/slide_aggregation")


class _SlideAggregator(HeatmapVisualizer):
    """Base class for slide-level prediction aggregators."""

    image_builder: InMemoryHeatmapAssembler
    partial_image_builder: functools.partial

    def __init__(self, save_dir: str, ground_truth_col: str | None) -> None:
        self.partial_image_builder = functools.partial(
            InMemoryHeatmapAssembler,
            vis_mode=IdentityMode,
            save_dir=save_dir,
            interpolation="nearest",
            compress_accumulator_array=True,
        )
        super().__init__(self.partial_image_builder, save_dir)
        self.ground_truth_col = ground_truth_col

    def on_test_dataloader_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        metadata: dict,
        dataloader_idx: int,
    ) -> None:
        self.image_builder = self.partial_image_builder(
            metadata=metadata, save_dir=self.save_dir
        )
        self.slide_name = metadata["slide_name"]

        if self.ground_truth_col is not None:
            if self.ground_truth_col not in metadata:
                raise ValueError(
                    f"Column {self.ground_truth_col} specified as ground truth does not exist in dataset."
                )

            self.slide_label = int(metadata[self.ground_truth_col])

    def aggregate_heatmap(
        self, mask_size: int, threshold: int | None = None
    ) -> tuple[float, float] | tuple[float, float, float]:
        # Sort out overlaps
        heatmap_accumulator = pyvips.Image.new_from_array(
            self.image_builder.heatmap_accumulator
        )
        overlap_counter = pyvips.Image.new_from_array(
            self.image_builder.patch_overlap_counter
        )
        heatmap = heatmap_accumulator / overlap_counter

        # Create averaging mask
        mask_arr = np.full((mask_size, mask_size), 1 / mask_size**2)
        mask = pyvips.Image.new_from_array(mask_arr)

        # Convolve the heatmap with mask and take maximum
        predicted_score = heatmap.max() / 255
        averaged_heatmap = heatmap.convf(mask)
        averaged_score = (
            averaged_heatmap.crop(
                mask_size - 1,
                mask_size - 1,
                averaged_heatmap.width - mask_size + 1,
                averaged_heatmap.height - mask_size + 1,
            ).max()
            / 255
        )

        if threshold is None:
            return predicted_score, averaged_score

        predicted_class = int(averaged_score >= threshold)

        return predicted_score, averaged_score, predicted_class

    def save_roc(self, fpr: list[float], tpr: list[float], mask_size: float) -> None:
        auc_score = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve - {mask_size}x{mask_size} mask")
        plt.legend(loc="lower right")
        save_path = f"{self.save_dir}/roc_curve_{mask_size}.jpg"
        plt.savefig(save_path, bbox_inches="tight")
        mlflow.log_artifact(save_path, self.save_dir)
        logger.debug(f"ROC curve for {mask_size}x{mask_size} mask saved")


class SlidePredictor(_SlideAggregator):
    """A callback that aggregates tile predictions into a slide prediction.

    Attributes:
        mask_size: size of the averaging mask (mask_size N creates a mask of size NxN)
        threshold: threshold from which (incl.) a slide is considered positive
        save_dir: directory to which the prediction table will be saved
        ground_truth_col: name of column in the dataset containing ground truth.
            If None, only predictions without metrics will be saved.
    """

    image_builder: InMemoryHeatmapAssembler
    partial_image_builder: functools.partial

    def __init__(
        self,
        mask_size: int,
        threshold: float,
        save_dir: str,
        ground_truth_col: str | None = None,
    ) -> None:
        super().__init__(save_dir, ground_truth_col)
        self.mask_size = mask_size
        self.threshold = threshold

        prediction_table_columns = [
            "slide_name",
            "predicted_class",
            "predicted_score",
            "averaged_score",
        ]
        if ground_truth_col is not None:
            prediction_table_columns.append("ground_truth")

        self.prediction_table = pd.DataFrame(columns=prediction_table_columns)

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        predicted_score, averaged_score, predicted_class = self.aggregate_heatmap(
            self.mask_size, self.threshold
        )

        prediction = [
            self.slide_name,
            predicted_class,
            predicted_score,
            averaged_score,
        ]

        mlflow.log_metric(f"test/{self.slide_name}/predicted_score", predicted_score)
        mlflow.log_metric(f"test/{self.slide_name}/averaged_score", averaged_score)
        mlflow.log_metric(f"test/{self.slide_name}/predicted_class", predicted_class)

        if self.ground_truth_col is not None:
            prediction.append(self.slide_label)
            mlflow.log_metric(f"test/{self.slide_name}/ground_truth", self.slide_label)
        self.prediction_table.loc[len(self.prediction_table)] = prediction

    def on_test_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        save_path = f"{self.save_dir}/prediction_table.parquet"

        if self.ground_truth_col is None:
            self.prediction_table.to_parquet(save_path, index=False)
            return

        self.prediction_table.to_parquet(save_path, index=False)
        mlflow.log_artifact(save_path, self.save_dir)

        tn, fp, fn, tp = confusion_matrix(
            self.prediction_table["ground_truth"],
            self.prediction_table["predicted_class"],
        ).ravel()
        mlflow.log_metric("test/accuracy", (tp + tn) / (tp + tn + fp + fn))
        mlflow.log_metric("test/true_positive", tp)
        mlflow.log_metric("test/true_negative", tn)
        mlflow.log_metric("test/false_positive", fp)
        mlflow.log_metric("test/false_negative", fn)
        mlflow.log_metric("test/sensitivity", tp / (tp + fn))
        mlflow.log_metric("test/specificity", tn / (tn + fp))
        mlflow.log_metric("test/precision", tp / (tp + fp))

        fpr, tpr, thresholds = roc_curve(
            self.prediction_table["ground_truth"],
            self.prediction_table["averaged_score"],
        )

        self.save_roc(fpr, tpr, self.mask_size)


class SlidePredictionOptimizer(_SlideAggregator):
    """A callback that finds the best mask size and threshold for slide-level prediction aggregation.

    Attributes:
        mask_size_min: minimal mask size to convolve the heatmap with (incl.)
        mask_size_max: maximal mask size to convolve the heatmap with (incl.)
        save_dir: directory to which the result metrics will be saved
        ground_truth_col: name of column in the dataset containing ground truth
    """

    image_builder: InMemoryHeatmapAssembler
    partial_image_builder: functools.partial

    def __init__(
        self,
        mask_size_min: int,
        mask_size_max: int,
        save_dir: str,
        ground_truth_col: str,
    ) -> None:
        super().__init__(save_dir, ground_truth_col)
        self.mask_size_min = mask_size_min
        self.mask_size_max = mask_size_max
        self.predicted_scores = pd.DataFrame(
            columns=[
                "slide_name",
                "ground_truth",
                *[f"{mask}x{mask}" for mask in range(mask_size_min, mask_size_max + 1)],
            ]
        )

    def on_test_dataloader_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        dataloader_idx: int,
    ) -> None:
        slide_scores = {"ground_truth": self.slide_label, "slide_name": self.slide_name}

        # Convolve slide with all mask sizes
        for mask_size in range(self.mask_size_min, self.mask_size_max + 1):
            _, averaged_score = self.aggregate_heatmap(mask_size)
            slide_scores[f"{mask_size}x{mask_size}"] = averaged_score

        self.predicted_scores.loc[len(self.predicted_scores)] = slide_scores

    def on_test_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        # Compute, graph and save ROC curve for all mask sizes
        for mask_size in range(self.mask_size_min, self.mask_size_max + 1):
            scores = self.predicted_scores[f"{mask_size}x{mask_size}"]
            fpr, tpr, thresholds = roc_curve(
                self.predicted_scores["ground_truth"], scores
            )
            self.save_roc(fpr, tpr, mask_size)

            roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
            save_path = f"{self.save_dir}/roc_data_{mask_size}.parquet"
            roc_data.to_parquet(save_path)
            mlflow.log_artifact(save_path, self.save_dir)
