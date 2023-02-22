import asyncio
import math
import time
from typing import Iterable, Optional

from cotrader.datasource.candles.candle_datasource import CandleDatasource
from cotrader.datasource.candles.indicators_datasource import Candle
from cotrader.utils.utils import interval_to_ms, normalize_timeframe


class DummyStream:
    interval: str | None = None

    async def subscribe(self, symbol, interval):
        self.interval = interval

    async def wait_for_candle(self):
        assert self.interval is not None

        interval_ms = interval_to_ms(self.interval)
        now = int(time.time() * 1000)
        next_ts = ((now // interval_ms) + 1) * interval_ms
        wait_time = (next_ts - now) / 1000
        await asyncio.sleep(wait_time)

        candles = list(
            DummyDatasource.get_instance().get_candles(
                symbol="DUMMY",
                interval=self.interval,
                start_time=next_ts,
                end_time=next_ts + interval_ms,
            )
        )

        return candles[0]

    async def stop(self):
        pass

    async def destroy(self):
        pass


# For testing purposes (to not get banned from binance)
class DummyDatasource(CandleDatasource):
    _instance: Optional["DummyDatasource"] = None

    @staticmethod
    def get_instance():
        if DummyDatasource._instance is None:
            DummyDatasource._instance = DummyDatasource()
        return DummyDatasource._instance

    def get_candles(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> Iterable[Candle]:
        start_time, end_time = normalize_timeframe(start_time, end_time, interval)
        interval_ms = interval_to_ms(interval)

        scale = (1 / 1000) * 0.01
        magnitude = 10.0
        bias = 1500.0

        high_phase = 1.0
        low_phase = 1.5
        shadow_magnitude = 2.0

        for i in range(start_time, end_time, interval_ms):
            candle = Candle(timestamp=i)

            old_close = (math.sin((i - interval_ms) * scale)) * magnitude + bias
            new_close = (math.sin(i * scale)) * magnitude + bias

            high = old_close + (math.sin(i * scale + high_phase) + 1) * shadow_magnitude
            low = old_close - (math.sin(i * scale + low_phase) + 1) * shadow_magnitude

            # Open price = close price of previous candle
            candle.set("open", old_close)
            # Close price = open price of next candle
            candle.set("close", new_close)

            candle.set("high", high)
            candle.set("low", low)
            candle.set("volume", 5000.0)

            yield candle

        # for i in range(start_time, end_time, interval_ms):
        #     idx = (i // interval_ms) % 3
        #     yield Candle(
        #         timestamp=i,
        #         # Open price = close price of previous candle
        #         open=(idx * 500 + bias),
        #         high=(idx * 500 + bias) + 100,
        #         low=(idx * 500 + bias) - 100,
        #         # Close price = open price of next candle
        #         close=(idx * 500 + bias) + 50,
        #         volume=400,
        #     )
