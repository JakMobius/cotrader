import time

import dateparser
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from cotrader.datasource.candles.binance_datasource import BinanceDatasource
from cotrader.datasource.candles.caching_datasource import CachingDatasource
from cotrader.datasource.candles.indicators_datasource import IndicatorsDatasource
from cotrader.datasource.tensor.candle_dataset import (
    CandleDataset,
    WrappedCandleDatasource,
)
from cotrader.profit_logger import ProfitLoggerCallback
from cotrader.utils.utils import interval_to_ms


class TrainCandleDatamodule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        binance_datasource = BinanceDatasource()
        caching_interface = CachingDatasource(binance_datasource)

        indicators_interface = IndicatorsDatasource.load(
            caching_interface, cfg.data.indicators
        )

        self.train_datasources = []
        self.validation_datasources = []

        for source in cfg.data.sources:
            timestamps = self.get_timestamps(source)

            print(f"Downloading the data for {source.symbol}, {source.interval}")
            data_controller = caching_interface.get_controller_for(
                source.symbol, source.interval
            )
            data_controller.populate_index(
                timestamps["train"][0],
                timestamps["train"][1],
                tqdm(desc=f"{source.symbol}, {source.interval} [train]"),
            )
            data_controller.populate_index(
                timestamps["validation"][0],
                timestamps["validation"][1],
                tqdm(desc=f"{source.symbol}, {source.interval} [validation]"),
            )

            self.train_datasources.append(
                WrappedCandleDatasource(
                    datasource=indicators_interface,
                    start_time=timestamps["train"][0],
                    end_time=timestamps["train"][1],
                    interval=source.interval,
                    symbol=source.symbol,
                )
            )
            self.validation_datasources.append(
                WrappedCandleDatasource(
                    datasource=indicators_interface,
                    start_time=timestamps["validation"][0],
                    end_time=timestamps["validation"][1],
                    interval=source.interval,
                    symbol=source.symbol,
                )
            )

        self.train_dataset = CandleDataset(
            self.train_datasources,
            cfg.data.sequence_length,
            cfg.data.input_features,
            cfg.data.output_features,
            progress="Building train dataset",
            normalize_input_mean=self.cfg.data.normalize_input_mean,
            normalize_output_mean=self.cfg.data.normalize_output_mean,
        )

        self.validation_dataset = CandleDataset(
            self.validation_datasources,
            cfg.data.sequence_length,
            cfg.data.input_features,
            cfg.data.output_features,
            progress="Building validation dataset",
            normalize_input_mean=self.cfg.data.normalize_input_mean,
            normalize_output_mean=self.cfg.data.normalize_output_mean,
        )

        self.train_dataset.print_statistics()
        self.validation_dataset.print_statistics()

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
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.training.dataloader_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return (
            torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.cfg.data.batch_size,
                num_workers=self.cfg.training.dataloader_workers,
                persistent_workers=True,
            ),
        )
