import math
from pathlib import Path

import pl_bolts
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from cotrader.models.model import Model


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        if batch_first:
            # Shape for batch_first: [1, max_len, d_model]
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            # Original shape: [max_len, 1, d_model]
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape:
               - If batch_first=True: ``[batch_size, seq_len, embedding_dim]``
               - If batch_first=False: ``[seq_len, batch_size, embedding_dim]``
        """
        if self.batch_first:
            x = x + self.pe[:, : x.size(1)]
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformer(Model):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        sequence_length: int,
        learning_rate: float,
        hidden_dim: int,
        feedforward_dim: int,
        n_head: int,
        dropout: float,
        num_encoder_layers: int,
        warmup_steps: int,
        cosine_annealing_steps: int,
    ):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.cosine_annealing_steps = cosine_annealing_steps

        self.input_projection = nn.Linear(input_features, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, batch_first=True)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, n_head, feedforward_dim, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.output_projection = nn.Linear(hidden_dim, output_features)

    def forward(self, x: torch.Tensor):
        """Forward pass"""
        assert len(x.shape) == 3
        assert x.shape[1] == self.sequence_length
        assert x.shape[2] == self.input_features

        x = self.input_projection(x)  # [batch_size, seq_length, d_model]

        # Apply positional encoding (now supports batch_first=True)
        x = self.pos_encoder(x)  # No transposing needed

        output = self.transformer_encoder(x)
        output = output[:, -1, :]  # Get the last token output [batch_size, d_model]

        output = self.output_projection(output)  # [batch_size, output_features]

        return output

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

        val_loss = self.loss_function(y_hat, y[:, -1, :])
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)

        return {"loss": val_loss, "y_hat": y_hat}

    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
            optimizer, self.warmup_steps, self.cosine_annealing_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def save(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        checkpoint = torch.load(
            filepath, weights_only=True, map_location=torch.device("cpu")
        )
        self.load_state_dict(checkpoint)
