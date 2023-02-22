import math
from abc import ABC
from typing import Any, Iterable, Optional

import torch
from tqdm import tqdm


class Multicandle:
    _indicators: dict
    _timestamp: torch.Tensor

    def __init__(
        self,
        size: Optional[int] = None,
        candle: Optional["Candle"] = None,
        candles: Optional[list["Candle"]] = None,
    ):
        if candle is not None:
            size = 1

        if candles is not None:
            size = len(candles)

        assert size is not None and size >= 0, "Size must be a non-negative integer"
        self._indicators = {}
        self._timestamp = torch.zeros([size], dtype=torch.int64)
        # self._timestamp = torch.zeros([size])

        if candles is not None:
            self.set_candles(candles)
            return

        if candle is not None:
            self.set_candle(0, candle)

    def get(self, name: str):
        if name in self._indicators:
            return self._indicators[name]
        if hasattr(self, name):
            return getattr(self, name)
        raise ValueError(f"Value {name} not found in candle or _indicators")

    def set(self, name: str, value: torch.Tensor):
        assert not torch.isnan(value).any()
        assert name != "timestamp"
        assert value.dtype == torch.float32
        assert len(value.shape) > 1
        assert value.shape[0] == self.timestamp.shape[0]
        self._indicators[name] = value

    def _init_indicator(self, name: str, sample: torch.Tensor | float):
        shape = sample.shape if isinstance(sample, torch.Tensor) else (1,)
        self._indicators[name] = torch.zeros(
            (self.timestamp.shape[0],) + shape, dtype=torch.float32
        )

    def set_candle(self, idx: int, candle: "Candle"):
        assert (
            idx < self._timestamp.shape[0]
        ), f"Index {idx} out of bounds for multicandle of length {len(self.timestamp)}"
        self.timestamp[idx] = candle.timestamp

        for k, v in candle._indicators.items():
            if k not in self._indicators:
                self._init_indicator(k, v)
            self._indicators[k][idx] = v

    def set_candles_batched(
        self,
        candles: Iterable["Candle"],
        offset: int = 0,
        progress: Optional[tqdm] = None,
        batch_size: int = 4096,
    ):
        i = offset
        batch = list()
        for candle in candles:
            batch.append(candle)
            if len(batch) < batch_size:
                continue
            self.set_candles(batch, offset=i)
            i += len(batch)
            if progress:
                progress.update(len(batch))
            batch.clear()
        if len(batch) > 0:
            self.set_candles(batch, offset=i)
            if progress:
                progress.update(len(batch))

    def set_candles(self, candles: Iterable["Candle"], offset: int = 0):
        timestamps = [candle.timestamp for candle in candles]
        candle_count = len(timestamps)
        self._timestamp[offset : offset + candle_count] = torch.tensor(
            timestamps, dtype=torch.int64
        )

        for k, v in next(iter(candles))._indicators.items():
            if k not in self._indicators:
                self._init_indicator(k, v)

            tensor = torch.tensor(
                [candle._indicators[k] for candle in candles], dtype=torch.float32
            )
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(-1)
            self._indicators[k][offset : offset + candle_count] = tensor

    def __iter__(self):
        for i in range(len(self.timestamp)):
            yield self[i]

    def __len__(self):
        return self.timestamp.shape[0]

    def __getitem__(self, idx: int) -> "Candle":
        assert idx < len(self.timestamp)
        result = Candle(int(self.timestamp[idx].item()))
        for k, v in self._indicators.items():
            result.set(k, v[idx])
        return result

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        assert isinstance(value, torch.Tensor), "timestamp must be a torch.Tensor"
        assert value.dtype == torch.int64, "timestamp tensor must have dtype=int64"
        self._timestamp = value


class Candle:
    _indicators: dict
    _timestamp: int

    def __init__(
        self,
        timestamp: int,
    ):
        assert isinstance(timestamp, int), "timestamp must be an int"
        self._indicators = {}
        self._timestamp = timestamp

    def serialize(self):
        indicators_serializable = {
            k: (
                v.tolist()
                if hasattr(v, "tolist")
                else (v.item() if hasattr(v, "item") else v)
            )
            for k, v in self._indicators.items()
        }
        return [
            self.timestamp,
            indicators_serializable,
        ]

    @staticmethod
    def deserialize(data: dict):
        candle = Candle(timestamp=data[0])
        for k, v in data[1].items():
            candle.set(k, v)
        return candle

    def get(self, name: str):
        if name in self._indicators:
            return self._indicators[name]
        raise ValueError(f"Value {name} not found in indicators")

    def set(self, name: str, value: Any):
        assert not math.isnan(value)
        assert name != "timestamp"
        assert isinstance(value, float) or (
            hasattr(value, "dtype") and value.dtype == torch.float32
        ), "value must be float32"
        self._indicators[name] = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.timestamp != other.timestamp:
            return False

        if set(self._indicators.keys()) != set(other._indicators.keys()):
            return False

        for key in self._indicators:
            if self._indicators[key] != other._indicators[key]:
                return False

        return True

    def __str__(self):
        indicators_str = ", ".join(f"{k}: {v}" for k, v in self._indicators.items())
        return f"Candle(timestamp={self.timestamp}, indicators={{ {indicators_str} }})"

    def __repr__(self):
        return self.__str__()

    def copy(self):
        new_candle = Candle(self.timestamp)
        new_candle._indicators = self._indicators.copy()
        return new_candle

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        assert isinstance(value, int), "timestamp must be an int"
        self._timestamp = value


class CandleDatasource(ABC):
    def get_candles(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> Iterable[Candle]:
        raise NotImplementedError()
