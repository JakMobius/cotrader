import json
import os
import shutil
import time
import unittest

from cotrader.datasource.candles.dummy_datasource import DummyDatasource
from cotrader.datasource.candles.indicators_datasource import (
    EMAIndicator,
    IndicatorsDatasource,
    SMAIndicator,
)


class TestDatabase(unittest.IsolatedAsyncioTestCase):
    def setup_database(self):
        # Remove test_database directory if it exists
        if os.path.exists("test_database"):
            shutil.rmtree("test_database")

        # Create a test database
        self.interface = IndicatorsDatasource(
            DummyDatasource(),
            [
                SMAIndicator(10),
                SMAIndicator(5),
                EMAIndicator(10),
                EMAIndicator(5),
            ],
        )

    def day_ts(self, day):
        return self.week_ago + self.day * day

    def setUp(self) -> None:
        # Current timestamp in milliseconds
        now = int(time.time() * 1000)
        self.day = int(86400000)

        # Timestamp week ago at 00:00 in milliseconds
        self.week_ago = now - (self.day * 7)
        self.week_ago = self.week_ago - (self.week_ago % self.day)

    @staticmethod
    def get_index(symbol, interval):
        index_path = os.path.join("test_database", symbol, interval, "index.json")
        with open(index_path) as f:
            return json.load(f)

    def verify_ema(self, period, index, candles):
        """
        Calculates and verified EMA[index] based on EMA[index - period].
        """

        ema_string = "ema" + str(period)
        ema_calculated = candles[index].get(ema_string)

        ema_coefficient = 2 / (period + 1)
        ema_expected = candles[index - period].get(ema_string)

        for i in range(index - period + 1, index + 1):
            ema_expected = candles[i].get("close") * ema_coefficient + ema_expected * (
                1 - ema_coefficient
            )

        self.assertAlmostEqual(ema_calculated, ema_expected, delta=0.01)

    def verify_sma(self, period, index, candles):
        """
        Verifies SMA indicator on the given index and period
        """
        sma_string = "sma" + str(period)
        sma_calculated = candles[index].get(sma_string)

        sma_expected = (
            sum(candles[i].get("close") for i in range(index - period + 1, index + 1))
            / period
        )

        self.assertAlmostEqual(sma_calculated, sma_expected, delta=0.01)

    async def test_download_basic(self):
        self.setup_database()

        millis_15m = 15 * 60 * 1000

        candles = list(
            self.interface.get_candles(
                "ETHUSDT", "15m", self.day_ts(0), self.day_ts(0) + millis_15m * 15
            )
        )

        self.verify_ema(10, 10, candles)
        self.verify_ema(5, 10, candles)

        self.verify_sma(10, 10, candles)
        self.verify_sma(5, 10, candles)


if __name__ == "__main__":
    unittest.main()
