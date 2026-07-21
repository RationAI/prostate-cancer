from torch import Tensor

from prostate_cancer.mil_model_base import ProstateCancerMILBase
from prostate_cancer.typing import MILModelOutput, SLLabeledBagOfTilesSampleBatch


class ProstateCancerClassicMIL(ProstateCancerMILBase):
    """Classic MIL: trained only on slide-level (SL) labels, no TL supervision."""

    def training_step(self, batch: SLLabeledBagOfTilesSampleBatch) -> Tensor:
        # bag ~ all embeddings from a single slide
        bags, sl_labels, _ = batch

        sl_outputs, _, _, _ = self(bags)
        loss = self.sl_criterion(sl_outputs, sl_labels)

        self.log("train/loss", loss, on_step=True, prog_bar=True, batch_size=len(bags))

        self.train_metrics_sl.update(sl_outputs, sl_labels)
        self.log_dict(
            self.train_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return loss

    def validation_step(self, batch: SLLabeledBagOfTilesSampleBatch) -> None:
        bags, sl_labels, _ = batch

        sl_outputs, _, _, _ = self(bags)
        loss = self.sl_criterion(sl_outputs, sl_labels)

        self.log("validation/loss", loss, prog_bar=True, batch_size=len(bags))

        self.val_metrics_sl.update(sl_outputs, sl_labels)
        self.log_dict(
            self.val_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

    def test_step(self, batch: SLLabeledBagOfTilesSampleBatch) -> MILModelOutput:  # type: ignore[override]
        bags, sl_labels, _ = batch

        sl_outputs, tl_outputs, mask, attention = self(bags)

        self.test_metrics_sl.update(sl_outputs, sl_labels)
        self.log_dict(
            self.test_metrics_sl, on_epoch=True, on_step=False, batch_size=len(bags)
        )

        return sl_outputs.sigmoid(), tl_outputs.sigmoid(), mask, attention
