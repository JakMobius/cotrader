import random
import time
from typing import List

import dateparser
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from cotrader.datasource.candles.caching_datasource import CachingDatasource
from cotrader.datasource.candles.indicators_datasource import IndicatorsDatasource
from cotrader.datasource.tensor.candle_dataset import (
    CandleDataset,
    WrappedCandleDatasource,
)
from cotrader.profit_logger import ProfitLoggerCallback
from cotrader.utils.utils import interval_to_ms


class ValidationCandleDatamodule(pl.LightningDataModule):
    def __init__(self, cfg, rng: random.Random):
        super().__init__()
        self.cfg = cfg

        self.train_datasources: List[WrappedCandleDatasource] = []
        self.validation_datasources: List[WrappedCandleDatasource] = []

        caching_datasource = CachingDatasource()
        datasource = IndicatorsDatasource.load(
            caching_datasource, self.cfg.data.indicators
        )

        available_candles = self.cfg.validation.available_candles
        train_size = self.cfg.validation.train_size
        val_size = self.cfg.validation.validation_size
        self.train_start = rng.randrange(0, available_candles - train_size - val_size)

        for source in self.cfg.data.sources:
            interval_ms = interval_to_ms(source.interval)
            end_time: str = source.upper_date
            interval: str = source.interval
            symbol: str = source.symbol

            parsed_end_time = dateparser.parse(end_time)
            assert parsed_end_time is not None
            end = int(parsed_end_time.timestamp() * 1000)
            start = int(end - available_candles * interval_ms)

            train_end = self.train_start + train_size
            val_end = train_end + val_size
            interval_ms = interval_to_ms(interval)

            self.train_datasources.append(
                WrappedCandleDatasource(
                    datasource=datasource,
                    start_time=start + self.train_start * interval_ms,
                    end_time=start + train_end * interval_ms,
                    interval=interval,
                    symbol=symbol,
                )
            )
            self.validation_datasources.append(
                WrappedCandleDatasource(
                    datasource=datasource,
                    start_time=start + train_end * interval_ms,
                    end_time=start + val_end * interval_ms,
                    interval=interval,
                    symbol=symbol,
                )
            )

        self.train_dataset = CandleDataset(
            self.train_datasources,
            self.cfg.data.sequence_length,
            self.cfg.data.input_features,
            self.cfg.data.output_features,
            normalize_input_mean=self.cfg.data.normalize_input_mean,
            normalize_output_mean=self.cfg.data.normalize_output_mean,
        )

        self.validation_dataset = CandleDataset(
            self.validation_datasources,
            self.cfg.data.sequence_length,
            self.cfg.data.input_features,
            self.cfg.data.output_features,
            normalize_input_mean=self.cfg.data.normalize_input_mean,
            normalize_output_mean=self.cfg.data.normalize_output_mean,
        )

    def create_profit_logger(self):
        return ProfitLoggerCallback(
            k=0.01,
            train_datasources=self.train_datasources,
            validation_datasources=self.validation_datasources,
            progress_bar=True,
        )

    def get_timestamps(self, cfg: DictConfig):
        end_time = cfg.training_end_time
        validation_candles = cfg.validation_candles
        train_candles = cfg.train_candles
        interval = cfg.interval

        interval_ms = interval_to_ms(interval)
        current_time = time.time() * 1000

        if end_time != "now":
            parsed_end_time = dateparser.parse(end_time)
            assert parsed_end_time is not None
            end = parsed_end_time.timestamp() * 1000
        else:
            end = current_time
        if end > current_time:
            end = current_time

        end -= end % interval_ms
        end = int(end)

        return {
            "train": (
                end - interval_ms * (train_candles + validation_candles),
                end - interval_ms * validation_candles,
            ),
            "validation": (end - interval_ms * validation_candles, end),
        }

    def train_dataloader(self):
        dataloader_workers = self.cfg.training.dataloader_workers
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=dataloader_workers,
            persistent_workers=dataloader_workers > 0,
        )

    def val_dataloader(self):
        dataloader_workers = self.cfg.training.dataloader_workers
        return (
            torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.cfg.data.batch_size,
                num_workers=dataloader_workers,
                persistent_workers=dataloader_workers > 0,
            ),
        )
