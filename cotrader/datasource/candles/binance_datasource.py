from typing import Iterable, Optional

from binance import (
    AsyncClient,
    BinanceSocketManager,
    Client,
    FuturesType,
    HistoricalKlinesType,
    ReconnectingWebsocket,
)

from cotrader.datasource.candles.candle_datasource import Candle, CandleDatasource
from cotrader.datasource.candles.dummy_datasource import DummyDatasource
from cotrader.utils.utils import interval_to_ms, normalize_timeframe


class BinanceTooEarlyError(ValueError):
    def __init__(self, actual: int, expected: int):
        super().__init__(
            f"Binance returned {actual} candles, but {expected} were expected "
            "(likely too early for this range)"
        )
        self.actual = actual
        self.expected = expected


class BinanceStream:
    client: AsyncClient
    socket_manager: BinanceSocketManager
    socket: Optional[ReconnectingWebsocket]

    def __init__(self):
        self.client = AsyncClient("", "")
        self.socket_manager = BinanceSocketManager(self.client)
        self.socket = None

    async def subscribe(self, symbol, interval):
        await self.stop()

        self.socket = self.socket_manager.kline_futures_socket(
            symbol=symbol, futures_type=FuturesType.USD_M, interval=interval
        )

        await self.socket.connect()

    async def wait_for_candle(self):
        assert self.socket is not None

        while True:
            res = await self.socket.recv()
            if "e" not in res or res["e"] != "continuous_kline":
                raise ValueError("Bad response from the binance socket", res)

            candle_data = res["k"]
            if candle_data["x"] is False:
                continue

            result = Candle(timestamp=int(candle_data["t"]))
            result.set("open", float(candle_data["o"]))
            result.set("high", float(candle_data["h"]))
            result.set("low", float(candle_data["l"]))
            result.set("close", float(candle_data["c"]))
            result.set("volume", float(candle_data["v"]))
            return result

    async def stop(self):
        if self.socket is not None:
            await self.socket_manager._stop_socket(self.socket)
            await self.socket.close()
            self.socket = None

    async def destroy(self):
        await self.stop()
        await self.client.close_connection()


class BinanceDatasource(CandleDatasource):
    client: Client | None = None

    def ensure_client(self):
        if self.client is None:
            # API key is not needed as we are only subscribing to public data
            # self.client = AsyncClient("", "")
            self.client = Client("", "")

    def dummy_data(self, symbol, interval, start_time, end_time) -> Iterable[Candle]:
        yield from DummyDatasource.get_instance().get_candles(
            symbol, interval, start_time, end_time
        )

    def get_candles(
        self, symbol: str, interval: str, start_time: int, end_time: int
    ) -> Iterable[Candle]:
        start_time, end_time = normalize_timeframe(start_time, end_time, interval)

        if symbol == "DUMMY":
            yield from self.dummy_data(symbol, interval, start_time, end_time)
            return

        self.ensure_client()
        assert self.client is not None

        raw_data = self.client.get_historical_klines(
            symbol,
            interval,
            start_time,
            end_time,
            klines_type=HistoricalKlinesType.FUTURES,
        )

        raw_data = [ohlcv_data[0:6] for ohlcv_data in raw_data]

        interval_ms = interval_to_ms(interval)
        expected_candles = (end_time - start_time) // interval_ms
        if len(raw_data) < expected_candles:
            raise BinanceTooEarlyError(len(raw_data), expected_candles)

        # Convert elements [:, 1:6] to float
        for ohlcv_data in raw_data[0:expected_candles]:
            volume = float(ohlcv_data[5])
            # Dirty hack: since volumes can be passed as relative values,
            # zeroes or small values can break the preprocessing.
            if volume <= 10:
                volume = 10

            candle = Candle(
                timestamp=int(ohlcv_data[0]),
            )

            candle.set("open", float(ohlcv_data[1]))
            candle.set("high", float(ohlcv_data[2]))
            candle.set("low", float(ohlcv_data[3]))
            candle.set("close", float(ohlcv_data[4]))
            candle.set("volume", float(volume))

            yield candle
