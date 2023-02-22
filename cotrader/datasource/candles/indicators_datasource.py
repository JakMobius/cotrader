import math
from abc import ABC
from collections import deque
from typing import Iterable

import hydra
from omegaconf import OmegaConf

from cotrader.datasource.candles.candle_datasource import Candle, CandleDatasource
from cotrader.utils.utils import interval_to_ms, normalize_timeframe


class Indicator(ABC):
    lookbehind: int
    name: str

    def __init__(self):
        self.name = "unnamed"
        self.lookbehind = 0

    def feed(self, candle: Candle):
        pass

    def apply(self, candle: Candle):
        pass


class EMAIndicator(Indicator):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.name = "ema" + str(self.period)
        self.multiplier = 2 / (self.period + 1)

        # Since EMA weights decay exponentially, it cannot be calculated
        # precisely, introduce tolerance that determines initial lookbehind
        # window.
        tolerance = 0.001
        self.lookbehind = math.ceil(math.log(tolerance) / math.log(1 - self.multiplier))
        self.ema = None

    def feed(self, candle: Candle):
        if self.ema is None:
            self.ema = candle.get("close")
        else:
            self.ema = candle.get("close") * self.multiplier + self.ema * (
                1 - self.multiplier
            )

    def apply(self, candle: Candle):
        candle.set(self.name, self.ema)


class SMAIndicator(Indicator):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.name = "sma" + str(self.period)
        self.lookbehind = self.period
        self.window = deque()
        self.sum = 0

    def feed(self, candle: Candle):
        self.sum += candle.get("close")
        self.window.append(candle.get("close"))
        if len(self.window) > self.period:
            self.sum -= self.window.popleft()

    def apply(self, candle: Candle):
        candle.set(self.name, self.sum / self.period)


class AverageVolume(Indicator):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.name = "volume" + str(self.period)
        self.lookbehind = self.period
        self.window = deque()
        self.sum = 0

    def feed(self, candle: Candle):
        self.sum += candle.get("volume")
        self.window.append(candle.get("volume"))
        if len(self.window) > self.period:
            self.sum -= self.window.popleft()

    def apply(self, candle: Candle):
        candle.set(self.name, self.sum / self.period)


class IndicatorsDatasource(CandleDatasource):
    indicators: dict[str, Indicator]

    def __init__(self, interface: CandleDatasource, indicators: Iterable[Indicator]):
        super().__init__()
        self.interface = interface
        self.indicators = dict()
        self.lookbehind = 0
        for indicator in indicators:
            self.add_indicator(indicator)

    def add_indicator(self, indicator: Indicator):
        assert indicator.name not in self.indicators
        self.indicators[indicator.name] = indicator
        self.lookbehind = max(self.lookbehind, indicator.lookbehind)

    def get_candles(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> Iterable[Candle]:
        interval_ms = interval_to_ms(interval)
        start_time, end_time = normalize_timeframe(start_time, end_time, interval)
        lookbehind_start_time = start_time - self.lookbehind * interval_ms

        # Few sanity checks, just to be safe

        candle_index = -self.lookbehind
        first_lookbehind_candle = None
        first_candle = None

        for candle in self.interface.get_candles(
            symbol, interval, lookbehind_start_time, end_time
        ):
            if candle_index + self.lookbehind == 0:
                first_lookbehind_candle = candle
            if candle_index == 0:
                first_candle = candle

            for indicator in self.indicators.values():
                if candle_index >= -indicator.lookbehind:
                    indicator.feed(candle)

                if candle_index >= 0:
                    indicator.apply(candle)

            if candle_index >= 0:
                yield candle

            candle_index += 1

        assert first_lookbehind_candle is not None
        assert first_candle is not None

        assert candle_index >= 0, (
            f"Expected at least {self.lookbehind} candles, but"
            f" got {candle_index + self.lookbehind}"
        )

        assert first_lookbehind_candle.timestamp == lookbehind_start_time, (
            f"Expected first lookbehind candle timestamp to be {lookbehind_start_time},"
            f" but got {first_lookbehind_candle.timestamp}"
        )

        assert first_candle.timestamp == start_time, (
            f"Expected first candle timestamp to be {start_time}, but"
            f" got {first_candle.timestamp}"
        )

    @classmethod
    def load(cls, interface, cfg):
        indicator_library = OmegaConf.load("configs/sections/indicators.yaml")

        indicators = []
        for indicator_name in cfg:
            if indicator_name in indicator_library:
                indicator_config = indicator_library[indicator_name]
                indicator = hydra.utils.instantiate(indicator_config)
                indicators.append(indicator)
            else:
                print(f"Warning: Indicator '{indicator_name}' not found in library")

        return IndicatorsDatasource(interface, indicators)
