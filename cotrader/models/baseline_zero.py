from pathlib import Path

import torch
import torch.nn as nn

from cotrader.models.model import Model


class BaselineZeroModel(Model):
    loss_function = nn.MSELoss()

    def __init__(
        self,
        input_features: int,
        output_features: int,
        sequence_length: int,
    ):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.sequence_length = sequence_length
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor):
        """Forward pass"""
        assert len(x.shape) == 3
        assert x.shape[1] == self.sequence_length
        assert x.shape[2] == self.input_features

        batch_size = x.shape[0]
        return torch.zeros(
            batch_size, self.output_features, device=x.device, dtype=x.dtype
        )

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y, *_ = batch

        self.sanitize_nans("x", x)
        self.sanitize_nans("y", y)

        y_hat = self(x)

        self.sanitize_nans("y_hat", y_hat, x)

        loss = self.loss_function(y_hat, y[:, -1, :])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return {"loss": loss, "y_hat": y_hat}

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y, *_ = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y[:, -1, :])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "y_hat": y_hat}

    def configure_optimizers(self):
        """Configure optimizers"""
        return []

    def save(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        checkpoint = torch.load(
            filepath, weights_only=True, map_location=torch.device("cpu")
        )
        self.load_state_dict(checkpoint)
