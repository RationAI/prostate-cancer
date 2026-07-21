from torch import Tensor

from prostate_cancer.mil_model_base import ProstateCancerMILBase
from prostate_cancer.typing import SLLabeledBagOfTilesSampleBatch


class ProstateCancerClassicMIL(ProstateCancerMILBase):
    """Classic MIL: trained only on slide-level (SL) labels, no TL supervision.

    `test_step` (SL + TL metrics) and the architecture are inherited unchanged
    from `ProstateCancerMILBase` - TL ground truth is still used to evaluate
    the (unsupervised) per-tile classifier at test time, it just never
    contributes to the training loss here.
    """

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
