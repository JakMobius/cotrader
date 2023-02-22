import unittest

from cotrader.datasource.candles.candle_datasource import Multicandle
from cotrader.datasource.candles.dummy_datasource import DummyDatasource


class TestDatabase(unittest.IsolatedAsyncioTestCase):
    def test_multicandle_set(self):
        multicandle = Multicandle(size=1024)
        datasource = DummyDatasource()

        candles = list(datasource.get_candles("DUMMY", "1s", 0 * 1000, 1024 * 1000))

        multicandle.set_candles([candles[0]], 0)
        assert candles[0] == multicandle[0]
        multicandle.set_candles([candles[0]], 1)
        assert candles[0] == multicandle[0]
        assert candles[0] == multicandle[1]

        multicandle.set_candles(candles, 0)

        for i in range(0, 1024):
            self.assertEqual(multicandle[i], candles[i], msg=f"candle #{i} is wrong")

        multicandle = Multicandle(size=2048)
        multicandle.set_candles(candles, 5)

        for i in range(0, 1024):
            self.assertEqual(
                multicandle[i + 5], candles[i], msg=f"candle #{i} is wrong"
            )

    def test_multicandle_set_batched(self):
        multicandle = Multicandle(size=1024)
        datasource = DummyDatasource()

        candles = list(datasource.get_candles("DUMMY", "1s", 0 * 1000, 1024 * 1000))

        multicandle.set_candles_batched([candles[0]], 0)
        assert candles[0] == multicandle[0]

        multicandle.set_candles_batched(candles, 0, batch_size=128)

        for i in range(0, 1024):
            self.assertEqual(multicandle[i], candles[i], msg=f"candle #{i} is wrong")

        multicandle = Multicandle(size=2048)
        multicandle.set_candles_batched(candles, 5, batch_size=128)

        for i in range(0, 1024):
            self.assertEqual(
                multicandle[i + 5], candles[i], msg=f"candle #{i} is wrong"
            )


if __name__ == "__main__":
    unittest.main()
