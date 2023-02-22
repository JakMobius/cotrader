from abc import ABC
from typing import Iterable, List, Optional

import pytorch_lightning as pl
import torch
from mlflow import MlflowClient
from PIL import Image, ImageDraw
from pytorch_lightning import Callback
from pytorch_lightning.loggers import MLFlowLogger
from tqdm import tqdm

from cotrader.datasource.candles.candle_datasource import Multicandle
from cotrader.datasource.tensor.candle_dataset import WrappedCandleDatasource
from cotrader.webserver.cost_predictor import CostPredictor


class Heatmap(ABC):
    heatmap: Optional[torch.Tensor] = None
    heatmap_dim: int
    plot_dim: int

    def __init__(self, heatmap_dim: int = 256, plot_dim: int = 1024):
        super().__init__()
        self.heatmap_dim = heatmap_dim
        self.plot_dim = plot_dim

    def get_type(self) -> str:
        raise NotImplementedError()

    def reset(self):
        self.heatmap = torch.zeros([self.heatmap_dim, self.heatmap_dim])

    def heat_color(self, value: float):
        value = 255 - value
        return (value, value, 255)

    def get_ticks(self):
        raise NotImplementedError()

    def project_profit(self, profit: torch.Tensor):
        raise NotImplementedError()

    def plot_values(
        self,
        old_candles: Multicandle,
        new_candles: Multicandle,
        real_candles: Multicandle,
    ):
        assert self.heatmap is not None

        predicted_gain = (
            new_candles._indicators["close"] / old_candles._indicators["close"]
        )
        true_gain = real_candles._indicators["close"] / old_candles._indicators["close"]

        indices_x = torch.round(self.project_profit(true_gain - 1) * self.heatmap_dim)
        indices_y = torch.round(
            self.project_profit(predicted_gain - 1) * self.heatmap_dim
        )

        indices_x = torch.clamp(indices_x, 0, self.heatmap_dim - 1)
        indices_y = torch.clamp(indices_y, 0, self.heatmap_dim - 1)

        indices_x = indices_x.type(torch.int32).flatten()
        indices_y = indices_y.type(torch.int32).flatten()

        self.heatmap[indices_x, indices_y] += 1

    def get_image(self):
        assert self.heatmap is not None

        max_value = torch.max(self.heatmap).item()
        map = torch.clamp(self.heatmap, 0, max_value) / max_value
        img = Image.new("RGB", (self.heatmap_dim, self.heatmap_dim), "white")
        pixels = (map * 255).to(torch.uint8).numpy()
        img.putdata([self.heat_color(v) for v in pixels.flatten()])

        img = img.resize((self.plot_dim, self.plot_dim), Image.Resampling.NEAREST)

        draw = ImageDraw.Draw(img)
        midpoint = self.plot_dim / 2

        draw.line(
            [(0, midpoint), (self.plot_dim, midpoint)],
            fill="black",
            width=2,
        )
        draw.line([(midpoint, 0), (midpoint, self.plot_dim)], fill="black", width=2)

        displayed_profits = self.get_ticks()
        line_points = self.project_profit(displayed_profits)

        line_points *= self.plot_dim

        fill = (80, 80, 80)

        for position, profit in zip(line_points.tolist(), displayed_profits):
            string = f"{profit.item() * 100:1g}%"
            if profit > 0:
                string = "+" + string

            draw.line(
                [(position, midpoint - 10), (position, midpoint + 10)],
                fill=fill,
                width=2,
            )

            draw.text(
                (position, midpoint - 15),
                align="center",
                anchor="mm",
                text=string,
                fill=fill,
            )

            draw.line(
                [(midpoint - 10, position), (midpoint + 10, position)],
                fill=fill,
                width=2,
            )

            draw.text(
                (midpoint + 15, position),
                align="center",
                anchor="lm",
                text=string,
                fill=fill,
            )

        return img


class LinearHeatmap(Heatmap):
    max_profit: float
    ticks: int

    def __init__(
        self,
        max_profit: float = 0.015,
        ticks: int = 5,
        heatmap_dim: int = 256,
        plot_dim: int = 1024,
    ):
        super().__init__(heatmap_dim=heatmap_dim, plot_dim=plot_dim)
        self.max_profit = max_profit
        self.ticks = ticks

    def get_ticks(self):
        return torch.tensor(
            list(
                map(
                    lambda idx: idx / self.ticks * self.max_profit,
                    range(-self.ticks, self.ticks + 1),
                )
            )
        )

    def project_profit(self, profit: torch.Tensor):
        return profit / self.max_profit / 2 + 0.5

    def get_type(self) -> str:
        return "linear"


class LogarithmicHeatmap(Heatmap):
    max_profit_power = -1
    min_profit_power = -6

    def __init__(
        self,
        max_profit_power: int = -1,
        min_profit_power: int = -6,
        heatmap_dim: int = 256,
        plot_dim: int = 1024,
    ):
        super().__init__(heatmap_dim=heatmap_dim, plot_dim=plot_dim)
        self.max_profit_power = max_profit_power
        self.min_profit_power = min_profit_power

    def get_ticks(self):
        powers = list(
            range(self.min_profit_power + 1, self.max_profit_power + 1),
        )
        return torch.cat(
            [
                10.0 ** torch.tensor(powers, dtype=torch.float32),
                -(10.0 ** torch.tensor(powers, dtype=torch.float32)),
            ]
        )

    def project_profit(self, profit: torch.Tensor):
        sign = profit < 0
        profit = torch.where(sign, -profit, profit)

        zero = profit < 10**self.min_profit_power

        result = (torch.log10(profit) - self.min_profit_power) / (
            self.max_profit_power - self.min_profit_power
        )
        result = torch.where(zero, 0, result)
        result = torch.where(sign, -result, result)

        return result / 2 + 0.5

    def get_type(self) -> str:
        return "logarithmic"


class ProfitLoggerCallback(Callback):
    predictor: Optional[CostPredictor] = None

    train_heatmaps: list[Heatmap] = [
        LinearHeatmap(max_profit=0.015, ticks=5),
        LogarithmicHeatmap(max_profit_power=-1, min_profit_power=-6),
    ]
    validation_heatmaps: list[Heatmap] = [
        LinearHeatmap(max_profit=0.015, ticks=5),
        LogarithmicHeatmap(max_profit_power=-1, min_profit_power=-6),
    ]

    def __init__(
        self,
        train_datasources: List[WrappedCandleDatasource],
        validation_datasources: List[WrappedCandleDatasource],
        k: float = 0.01,  # every position is 1% of deposit by default
        progress_bar: bool = True,
    ):
        super().__init__()
        self.position_scale = k
        self.train_datasources = train_datasources
        self.validation_datasources = validation_datasources
        # TODO: bad
        self.interval_ms = self.train_datasources[0].interval_ms

        self.train_candles = Multicandle(
            size=sum([len(source) for source in train_datasources])
        )
        self.validation_candles = Multicandle(
            size=sum([len(source) for source in validation_datasources])
        )

        self.build_candles(
            train_datasources,
            self.train_candles,
            progress_desc=(
                "Building ProfitLoggerCallback train candles" if progress_bar else None
            ),
        )
        self.build_candles(
            validation_datasources,
            self.validation_candles,
            progress_desc=(
                "Building ProfitLoggerCallback validation candles"
                if progress_bar
                else None
            ),
        )

    def set_predictor(self, predictor: CostPredictor):
        self.predictor = predictor
        assert (
            self.predictor.get_predictor_lookbehind() == 0
        ), "Predictor can't look too far behind"

    def build_candles(
        self,
        datasources: List[WrappedCandleDatasource],
        multicandle: Multicandle,
        progress_desc: Optional[str],
    ):
        progress = (
            tqdm(
                desc=progress_desc,
                total=sum(len(ds) for ds in datasources),
                unit="candle",
            )
            if progress_desc
            else None
        )

        i = 0
        for datasource in datasources:
            multicandle.set_candles_batched(datasource, i, progress)
            i += len(datasource)

        if progress:
            progress.close()

    def get_trailing_candle_mask(
        self,
        seq_end_idx: torch.Tensor,
        datasources: List[WrappedCandleDatasource],
    ):
        result = torch.full_like(seq_end_idx, True, dtype=torch.bool)

        start_index = 0
        for i, datasource in enumerate(datasources):
            end_index = start_index + len(datasource)
            result = torch.where(seq_end_idx == end_index - 1, False, result)
            start_index = end_index

        return result

    def get_predictions(
        self,
        seq_start_idx: torch.Tensor,
        outputs: torch.Tensor,
        dataset: Multicandle,
        sequence_length: int,
    ):
        next_candle_idx = seq_start_idx + sequence_length
        last_candle_idx = next_candle_idx - 1
        batch_size = len(seq_start_idx)

        old_candles = Multicandle(size=batch_size)
        new_candles = Multicandle(size=batch_size)
        real_candles = Multicandle(size=batch_size)

        for feature in dataset._indicators.keys():
            shape = dataset._indicators[feature].shape[1:]
            old_candles._indicators[feature] = dataset._indicators[feature][
                last_candle_idx
            ]
            real_candles._indicators[feature] = dataset._indicators[feature][
                next_candle_idx
            ]
            new_candles._indicators[feature] = torch.zeros((batch_size,) + shape)

        old_candles.timestamp = dataset.timestamp[last_candle_idx]
        real_candles.timestamp = dataset.timestamp[next_candle_idx]
        new_candles.timestamp = dataset.timestamp[next_candle_idx]

        assert (
            real_candles.timestamp == old_candles.timestamp + self.interval_ms
        ).all()

        def window_accessor(idx: int):
            if idx == 0:
                return new_candles
            if idx == -1:
                return old_candles
            assert False, "Predictor can't look too far behind"

        assert self.predictor is not None

        # Interval does not matter here, the important bit is whether
        # the candle goes up or down
        self.predictor.extract_candle_from_prediction(
            window_accessor, new_candles, outputs, "1m"
        )

        return (old_candles, new_candles, real_candles)

    def log_profits(
        self,
        metric: str,
        old_candles: Multicandle,
        new_candles: Multicandle,
        real_candles: Multicandle,
        pl_module: pl.LightningModule,
    ):
        batch_size = len(old_candles)

        predicted_gain = (
            new_candles._indicators["close"] / old_candles._indicators["close"]
        ).flatten()
        true_gain = (
            real_candles._indicators["close"] / old_candles._indicators["close"]
        ).flatten()

        buy = predicted_gain > 1
        correct_buy = true_gain > 1

        mistakes = torch.count_nonzero(correct_buy ^ buy)
        correct_ratio = 1 - mistakes / batch_size

        profits = (
            torch.where(buy, true_gain, 1 / true_gain) - 1
        ) * self.position_scale + 1

        batch_profit = torch.prod(profits)
        batch_time = batch_size * self.interval_ms
        year = 365 * 24 * 60 * 60 * 1000
        annualized_profit = batch_profit ** (year / batch_time)

        pl_module.log(
            f"{metric}_correct_rate", correct_ratio, on_epoch=True, on_step=False
        )
        pl_module.log(
            f"{metric}_annual_profit",
            annualized_profit.item(),
            on_epoch=True,
            on_step=False,
        )

    def get_mlflow_loggers(self, pl_module: pl.Trainer) -> Iterable[MLFlowLogger]:
        loggers = pl_module.loggers

        for logger in loggers:
            if isinstance(logger, MLFlowLogger):
                yield logger

        return None

    def handle_batch_end(
        self,
        name: str,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        candles: Multicandle,
        heatmaps: List[Heatmap],
    ):
        x, _, seq_start_idx, timestamps = batch
        sequence_len: int = x.shape[1]
        batch_size: int = x.shape[0]

        assert isinstance(outputs, dict)
        predictions = outputs["y_hat"]
        assert isinstance(seq_start_idx, torch.Tensor)
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(timestamps, torch.Tensor)
        assert len(predictions.shape) == 2
        assert predictions.shape[0] == batch_size

        old_candles, new_candles, real_candles = self.get_predictions(
            seq_start_idx.to("cpu"),
            predictions,
            candles,
            sequence_len,
        )

        expected_timestamps = (
            timestamps.to("cpu") + (sequence_len - 1) * self.interval_ms
        )
        if (old_candles.timestamp != expected_timestamps).any():
            print("Timestamps mismatch!")
            print("by index", old_candles.timestamp.tolist())
            print("by batch", expected_timestamps.tolist())
            print("seq_starts", seq_start_idx.tolist())
            mismatch_indices = (old_candles.timestamp != expected_timestamps).nonzero(
                as_tuple=True
            )[0]
            print("Mismatch at indices:", mismatch_indices.tolist())
            for idx in mismatch_indices.tolist():
                print(
                    f"Index {idx}: by index = {old_candles.timestamp[idx]}, from batch"
                    f" = {expected_timestamps[idx]}"
                )
            assert False

        self.log_profits(
            metric=name,
            new_candles=new_candles,
            old_candles=old_candles,
            real_candles=real_candles,
            pl_module=pl_module,
        )

        for map in heatmaps:
            map.plot_values(
                new_candles=new_candles,
                old_candles=old_candles,
                real_candles=real_candles,
            )

    def handle_epoch_end(self, name: str, trainer: pl.Trainer, heatmaps: List[Heatmap]):
        for logger in self.get_mlflow_loggers(trainer):
            client: MlflowClient = logger.experiment
            assert logger.run_id is not None
            for map in heatmaps:
                client.log_image(
                    logger.run_id,
                    image=map.get_image(),
                    key=f"heatmap/{name}/{map.get_type()}/",
                    step=trainer.global_step,
                )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self.handle_batch_end(
            "train",
            pl_module,
            outputs,
            batch,
            self.train_candles,
            self.train_heatmaps,
        )

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        for heatmap in self.train_heatmaps:
            heatmap.reset()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.handle_epoch_end("train", trainer, self.train_heatmaps)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.handle_batch_end(
            "validation",
            pl_module,
            outputs,
            batch,
            self.validation_candles,
            self.validation_heatmaps,
        )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        for heatmap in self.validation_heatmaps:
            heatmap.reset()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.handle_epoch_end("validation", trainer, self.validation_heatmaps)
