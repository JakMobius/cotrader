from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from cotrader.models.model import Model


class IntermediadeBlock(pl.LightningModule):
    def __init__(self, hidden_dim: int, dropout: float, activation: type[nn.Module]):
        super().__init__()
        self.lhs = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )

        self.rhs = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.LayerNorm([hidden_dim]),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
        )

        self.norm = nn.LayerNorm([hidden_dim])
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lhs = self.lhs(input)
        rhs = self.rhs(input)

        lhs = self.norm(lhs + input)

        cat = torch.cat([lhs, rhs], dim=-1)

        return self.out(cat)


class LinearV2(Model):
    input_layer: nn.Linear
    blocks: List[IntermediadeBlock]
    output_layer: nn.Linear

    def __init__(
        self,
        input_features: int,
        output_features: int,
        sequence_length: int,
        dropout: float,
        hidden_layers: int,
        hidden_dim: int,
        learning_rate: float,
    ):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        self.activation: type[nn.Module] = nn.GELU

        layers: list[nn.Module] = [nn.Flatten()]
        layers.append(nn.Linear(sequence_length * input_features, hidden_dim))

        for i in range(hidden_layers):
            layers.append(
                IntermediadeBlock(
                    dropout=dropout, hidden_dim=hidden_dim, activation=self.activation
                )
            )

        layers.append(nn.Linear(hidden_dim, output_features))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Forward pass"""
        assert len(x.shape) == 3
        assert x.shape[1] == self.sequence_length
        assert x.shape[2] == self.input_features

        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values.")

        # x shape: [batch, seq_len, features]
        x = x.flatten(start_dim=1)
        return self.model(x)

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
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def save(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        checkpoint = torch.load(
            filepath, weights_only=True, map_location=torch.device("cpu")
        )
        self.load_state_dict(checkpoint)
