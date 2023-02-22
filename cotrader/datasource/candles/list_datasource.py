from typing import Iterator, Sequence

from cotrader.datasource.candles.candle_datasource import CandleDatasource
from cotrader.datasource.candles.indicators_datasource import Candle
from cotrader.utils.utils import interval_to_ms, normalize_timeframe


class ListDatasource(CandleDatasource):
    def __init__(self, seq: Sequence[Candle]):
        self.seq = seq

    def get_candles(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> Iterator[Candle]:
        start_time, end_time = normalize_timeframe(start_time, end_time, interval)
        interval_ms = interval_to_ms(interval)

        if self.seq[0].timestamp > start_time:
            raise ValueError(
                f"start_time={start_time} out of bounds. Available timeframe: "
                f"{self.seq[0].timestamp} - {self.seq[-1].timestamp}"
            )

        if self.seq[-1].timestamp < end_time - interval_ms:
            raise ValueError(
                f"end_time={end_time} out of bounds. Available timeframe: "
                f"{self.seq[0].timestamp} - {self.seq[-1].timestamp}"
            )

        start_index = (start_time - self.seq[0].timestamp) // interval_ms
        end_index = (end_time - self.seq[0].timestamp) // interval_ms

        for i in range(start_index, end_index):
            yield self.seq[i]
