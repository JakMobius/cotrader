from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from cotrader.models.model import Model


class LightningLSTMModel(Model):
    """
    LSTM model for time series prediction using PyTorch Lightning.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        sequence_length: int,
        hidden_dim: int,
        learning_rate: float,
        hidden_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        # Define model
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, output_features)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)
        # Use only the last output
        out = self.fc(lstm_out)
        return out

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "y_hat": y_hat}

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "y_hat": y_hat}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def save(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        checkpoint = torch.load(
            filepath, map_location=torch.device("cpu"), weights_only=True
        )
        self.load_state_dict(checkpoint)
