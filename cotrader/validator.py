import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import dateparser
import hydra
import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping
from torch.multiprocessing import Pool
from tqdm import tqdm

from cotrader.datasource.candles.binance_datasource import BinanceDatasource
from cotrader.datasource.candles.caching_datasource import CachingDatasource
from cotrader.datasource.tensor.validation_candle_datamodule import (
    ValidationCandleDatamodule,
)
from cotrader.profit_logger import ProfitLoggerCallback
from cotrader.train_utils import get_trainer_loggers
from cotrader.utils.utils import interval_to_ms
from cotrader.webserver.fixed_sequence_predictor import FixedSequencePredictor


class Validator:
    profit_logging_callback: ProfitLoggerCallback

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        """Main method to start training."""

        self.find_unique_experiment_dir()
        self.download_data()

        self.metrics = {}
        self.train_start_idx = {}

        if "histograms" in self.cfg.output:
            matplotlib.use("Agg")

        printed_weights = False

        for metrics, i, train_start_idx, weights, time_used in self.iter_losses():
            if not printed_weights:
                print(f"Model weights count: {weights}")
            print(
                f"[- step #{i}] time: {time_used:.2f}sec, "
                "losses: "
                f"train={metrics['train_loss']:.6f}, "
                f"val={metrics['val_loss']:.6f}"
            )

            self.metrics[i] = metrics
            self.train_start_idx[i] = train_start_idx
            self.log(weights)

        (Path(self.experiment_dir) / "config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg)
        )

    def get_statistics(self, metric: dict):
        tensor = torch.tensor(list(metric.items()))
        max_idx = int(tensor[:, 1].argmax().item())
        min_idx = int(tensor[:, 1].argmin().item())
        return {
            "max_idx": max_idx,
            "max": tensor[max_idx, 1].item(),
            "min_idx": min_idx,
            "min": tensor[min_idx, 1].item(),
            "median": tensor[:, 1].median().item(),
            "avg": tensor[:, 1].mean().item(),
        }

    def log(self, weights):
        epochs = list(self.metrics.keys())
        keys = self.metrics[epochs[0]].keys()

        serialized_metrics = {
            key: self.get_statistics(
                {epoch: self.metrics[epoch][key] for epoch in epochs}
            )
            for key in keys
        }

        result_stats = json.dumps(
            {
                "progress": f"{len(self.metrics)} / {self.cfg.validation.steps}",
                "weights": weights,
                "metrics": serialized_metrics,
                "log": self.metrics,
            },
            indent=4,
        )

        (Path(self.experiment_dir) / "result.json").write_text(result_stats)

        if "histograms" in self.cfg.output:
            self.draw_histograms()

    def draw_histograms(self):
        def plot(file, metrics, name, xlabel, ylabel):
            for metric in metrics:
                values = [self.metrics[epoch][metric] for epoch in self.metrics.keys()]
                plt.hist(
                    values,
                    bins=30,
                    alpha=0.7,
                    label=metric,
                )
            plt.title(name)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.savefig((Path(self.experiment_dir) / file).absolute())
            plt.close()

        for histogram in self.cfg.output.histograms:
            plot(
                file=histogram.file,
                metrics=histogram.metrics,
                name=histogram.name,
                xlabel=histogram.xlabel,
                ylabel=histogram.ylabel,
            )

    def iter_losses(self):
        steps = self.cfg.validation.steps
        if self.cfg.validation.nproc > 1:
            pool = Pool(self.cfg.validation.nproc)
            yield from pool.imap_unordered(
                self.train_task.__get__(self, Validator), range(steps)
            )
            pool.close()
            pool.join()
        else:
            for i in range(steps):
                yield self.train_task(i)

    def download_data(self):
        binance_interface = BinanceDatasource()
        caching_datasource = CachingDatasource(binance_interface)
        available_candles = self.cfg.validation.available_candles

        for source in self.cfg.data.sources:
            interval_ms = interval_to_ms(source.interval)
            end_time: str = source.upper_date
            interval: str = source.interval
            symbol: str = source.symbol

            parsed_end_time = dateparser.parse(end_time)
            assert parsed_end_time is not None
            end = int(parsed_end_time.timestamp() * 1000)
            # TODO: use feature padding
            padding = 1024
            start = int(end - (available_candles + padding) * interval_ms)

            caching_datasource.get_controller_for(symbol, interval).populate_index(
                start, end, tqdm(desc=f"Downloading {symbol}", unit="candle")
            )

    def find_unique_experiment_dir(self):
        base_dir = self.cfg.output.experiment_dir
        num = 0
        unique_dir = f"{base_dir}-{num}"
        while os.path.exists(unique_dir):
            num += 1
            unique_dir = f"{base_dir}-{num}"
        self.experiment_dir = unique_dir

    def train_task(self, step):
        start_time = time.time()
        seed = random.Random(x=self.cfg.validation.seed).getrandbits(32) + step
        rng = random.Random(seed)

        disable_output = self.cfg.validation.suppress_output
        if disable_output:
            sys.stdout = sys.stderr = open(os.devnull, "w")
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
                logging.WARNING
            )

        datamodule = ValidationCandleDatamodule(self.cfg, rng)

        model = hydra.utils.instantiate(
            self.cfg.model,
            input_features=sum(
                f.get_feature_count() for f in datamodule.train_dataset.input_features
            ),
            output_features=sum(
                f.get_feature_count() for f in datamodule.train_dataset.output_features
            ),
        )

        metrics = self.train_model(model, datamodule, step)

        return (
            metrics,
            step,
            datamodule.train_start,
            sum(p.numel() for p in model.parameters()),
            time.time() - start_time,
        )

    def train_model(self, model, datamodule, step):
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
            callback = datamodule.create_profit_logger()
            callback.set_predictor(FixedSequencePredictor(self.cfg, model=model))
            callbacks.append(callback)

        if self.cfg.training.loss_function == "mae":
            model.loss_function = torch.nn.L1Loss()
        if self.cfg.training.loss_function == "mse":
            model.loss_function = torch.nn.MSELoss()

        trainer = pl.Trainer(
            max_epochs=self.cfg.training.epochs,
            logger=get_trainer_loggers(self.cfg, experiment_dir=self.experiment_dir),
            callbacks=callbacks,
            accelerator=self.cfg.device,
            devices=1,
            gradient_clip_val=self.cfg.training.gradient_clip_val,
            enable_progress_bar=not self.cfg.validation.suppress_output,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            enable_model_summary=not self.cfg.validation.suppress_output,
        )

        trainer.fit(model=model, datamodule=datamodule)

        return {
            k: v.item() if hasattr(v, "item") else v
            for k, v in trainer.callback_metrics.items()
        }
