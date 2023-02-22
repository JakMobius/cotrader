#!/usr/bin/env python3

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from cotrader.datasource.tensor.train_candle_datamodule import TrainCandleDatamodule
from cotrader.models.model import load_model, save_model
from cotrader.profit_logger import ProfitLoggerCallback
from cotrader.train_utils import get_trainer_loggers
from cotrader.webserver.fixed_sequence_predictor import FixedSequencePredictor


class Trainer:
    profit_logging_callback: ProfitLoggerCallback

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        """Main method to start training."""

        cfg = self.cfg

        self.datamodule = TrainCandleDatamodule(cfg)

        self.model = load_model(cfg.model, self.datamodule.train_dataset)

        if cfg.training.load_checkpoint is True:
            self.model.load(Path(cfg.output.model_file) / "model.pt")

        self.train_model()
        save_model(self.model, cfg, cfg.output.model_file)

        print(f"Training completed. The model file is {cfg.output.model_file}")

    def train_model(self):
        """Trains the model using PyTorch Lightning."""
        callbacks = []

        if self.cfg.training.early_stopping_patience > 0:
            early_stopping_callback = EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.training.early_stopping_patience,
                mode="min",
                verbose=True,
            )
            callbacks.append(early_stopping_callback)

        if self.cfg.training.log_profit:
            callback = self.datamodule.create_profit_logger()
            callback.set_predictor(FixedSequencePredictor(self.cfg, model=self.model))
            callbacks.append(callback)

        callbacks.append(LearningRateMonitor())
        if self.cfg.training.checkpoints.enabled:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.cfg.training.checkpoints.dir,
                    filename=self.cfg.training.checkpoints.filename,
                    save_top_k=self.cfg.training.checkpoints.save_top_k,
                    monitor="val_loss",
                )
            )

        trainer = pl.Trainer(
            max_epochs=self.cfg.training.epochs,
            logger=get_trainer_loggers(self.cfg),
            callbacks=callbacks,
            accelerator=self.cfg.device,
            devices=1,
            gradient_clip_val=self.cfg.training.gradient_clip_val,
            enable_progress_bar=True,
        )

        print(f"Starting training for {self.cfg.training.epochs} epochs...")
        trainer.fit(model=self.model, datamodule=self.datamodule)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    Trainer(cfg).run()
    return 0


if __name__ == "__main__":
    train()
