import os
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy


class LitModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 10,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )

        # accuracy
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        out = self.model(x)
        return out

    def model_step(
        self, batch, stage=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds, y)

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/acc", acc, prog_bar=True)

        return loss, preds, y

    def training_step(self, batch, batch_idx) -> dict:
        loss, preds, targets = self.model_step(batch, "train")

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        self.model_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.model_step(batch, "test")

    def configure_optimizers(self) -> dict:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
