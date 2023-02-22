import json
import os
import shutil
import time
import unittest

from cotrader.datasource.candles.caching_datasource import CachingDatasource
from cotrader.datasource.candles.dummy_datasource import DummyDatasource
from cotrader.utils.utils import interval_to_ms

DAY_MILLIS = int(86400000)


class TestDatabase(unittest.IsolatedAsyncioTestCase):
    def setup_database(self):
        # Remove test_database directory if it exists
        if os.path.exists("test_database"):
            shutil.rmtree("test_database")

        # Create a test database
        self.cachingInterface = CachingDatasource(
            interface=DummyDatasource(), database_dir="test_database"
        )
        self.cachingInterface.consolidate = False

    def setUp(self) -> None:
        # Current timestamp in milliseconds
        now = int(time.time() * 1000)

        # Timestamp week ago at 00:00 in milliseconds
        self.week_ago = now - (DAY_MILLIS * 7)
        self.week_ago = self.week_ago - (self.week_ago % DAY_MILLIS)

    def tearDown(self) -> None:
        # Remove test_database directory if it exists
        if os.path.exists("test_database"):
            shutil.rmtree("test_database")
        pass

    @staticmethod
    def get_index(symbol, interval):
        index_path = os.path.join("test_database", symbol, interval, "index.json")
        with open(index_path) as f:
            return json.load(f)

    def verify_data(self, data, symbol, interval, time_from, time_to):
        assert self.cachingInterface.interface is not None

        original_data = list(
            self.cachingInterface.interface.get_candles(
                symbol, interval, time_from, time_to
            )
        )

        assert len(original_data) == (time_to - time_from) / interval_to_ms(interval)

        self.assertEqual(data, original_data)

    def verify_index_json(self, symbol, interval, timeframes):
        index = self.get_index(symbol, interval)
        self.assertEqual(
            [
                {
                    "from": timeframe[0],
                    "to": timeframe[1],
                    "file": CachingDatasource.filename_for_timeframe(
                        timeframe[0], timeframe[1]
                    ),
                }
                for timeframe in timeframes
            ],
            index,
        )

    def day_ts(self, day_index):
        return self.week_ago + DAY_MILLIS * day_index

    def test_download_basic(self):
        self.setup_database()

        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(0), self.day_ts(1)
            )
        )
        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(2), self.day_ts(3)
            )
        )

        # Ensure that the index.json file is correct. It should look like this:
        self.verify_index_json(
            "ETHUSDT",
            "15m",
            [
                [self.day_ts(0), self.day_ts(1)],
                [self.day_ts(2), self.day_ts(3)],
            ],
        )

        # Ensure that the data files are correct

        for filename in [
            CachingDatasource.filename_for_timeframe(self.day_ts(0), self.day_ts(1)),
            CachingDatasource.filename_for_timeframe(self.day_ts(2), self.day_ts(3)),
        ]:
            # Get the data file at test_database/ETHUSDT/15m/(filename)
            data_path = os.path.join("test_database", "ETHUSDT", "15m", filename)

            # Load the data from the file
            data = CachingDatasource.load_candles(data_path)

            # Ensure that the length of the data is correct
            self.assertEqual(len(data), DAY_MILLIS // (15 * 60 * 1000))

        # Ensure that the "find_nearest_left_index" function works correctly
        controller = self.cachingInterface.get_controller_for("ETHUSDT", "15m")

        assert controller.find_nearest_left_index(self.day_ts(0) - 1) == -1
        assert controller.find_nearest_left_index(self.day_ts(1) + 1) == 0
        assert controller.find_nearest_left_index(self.day_ts(2) - 1) == 0
        assert controller.find_nearest_left_index(self.day_ts(3) + 1) == 1

        # Ensure that the "find_nearest_time" function works correctly

        assert controller.find_nearest_time(self.day_ts(1) + 1, 0) == self.day_ts(1)
        assert (
            controller.find_nearest_time(self.day_ts(1) + 1000, self.day_ts(1) + 1)
            == self.day_ts(1) + 1
        )
        assert controller.find_nearest_time(self.day_ts(0) - 1, 100) == 100
        assert controller.find_nearest_time(100, self.day_ts(0) + 1) == self.day_ts(0)
        assert controller.find_nearest_time(100, 200) == 200
        assert controller.find_nearest_time(
            self.day_ts(2) - 1, self.day_ts(2) + 1
        ) == self.day_ts(2)

        # Edge case: week_ago + day is the right boundary of the first entry
        assert (
            controller.find_nearest_time(self.day_ts(1), self.day_ts(1) + 1)
            == self.day_ts(1) + 1
        )
        assert controller.find_nearest_time(self.day_ts(1), 0) == self.day_ts(1)

    def test_download_intersect(self):
        self.setup_database()

        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(0), self.day_ts(1)
            )
        )
        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(2), self.day_ts(3)
            )
        )

        intersecting_data_from = self.day_ts(0) + DAY_MILLIS // 2
        intersecting_data_to = self.day_ts(3) + DAY_MILLIS // 2

        # Request data that intersects with the both existing data
        intersecting_data = list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", intersecting_data_from, intersecting_data_to
            )
        )

        # Get index.json at test_database/ETHUSDT/15m/index.json
        self.verify_index_json(
            "ETHUSDT",
            "15m",
            [
                [self.day_ts(0), self.day_ts(1)],
                [self.day_ts(1), self.day_ts(2)],
                [self.day_ts(2), self.day_ts(3)],
                [self.day_ts(3), self.day_ts(3) + DAY_MILLIS // 2],
            ],
        )

        # Verify that the intersecting data is correct
        self.verify_data(
            intersecting_data,
            "ETHUSD",
            "15m",
            intersecting_data_from,
            intersecting_data_to,
        )

    def test_download_nested(self):
        self.setup_database()

        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(0), self.day_ts(1)
            )
        )
        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(4), self.day_ts(5)
            )
        )

        nested_data_from = self.day_ts(2)
        nested_data_to = self.day_ts(3)

        # Request data that lays between the existing data
        nested_data = list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", nested_data_from, nested_data_to
            )
        )

        self.verify_index_json(
            "ETHUSDT",
            "15m",
            [
                [self.day_ts(0), self.day_ts(1)],
                [self.day_ts(2), self.day_ts(3)],
                [self.day_ts(4), self.day_ts(5)],
            ],
        )

        # Verify that the intersecting data is correct
        self.verify_data(nested_data, "ETHUSD", "15m", nested_data_from, nested_data_to)

    def test_download_adjacent(self):
        self.setup_database()

        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(0), self.day_ts(1)
            )
        )
        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(2), self.day_ts(3)
            )
        )

        adjacent_data_from = self.day_ts(1)
        adjacent_data_to = self.day_ts(2)

        # Request data that is adjacent to the existing data
        adjacent_data = list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", adjacent_data_from, adjacent_data_to
            )
        )

        self.verify_index_json(
            "ETHUSDT",
            "15m",
            [
                [self.day_ts(0), self.day_ts(1)],
                [self.day_ts(1), self.day_ts(2)],
                [self.day_ts(2), self.day_ts(3)],
            ],
        )

        # Verify that the intersecting data is correct
        self.verify_data(
            adjacent_data,
            "ETHUSD",
            "15m",
            adjacent_data_from,
            adjacent_data_to,
        )

    def test_download_consolidate(self):
        self.setup_database()
        self.cachingInterface.consolidate = True

        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(0), self.day_ts(1)
            )
        )
        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(2), self.day_ts(3)
            )
        )
        list(
            self.cachingInterface.get_candles(
                "ETHUSDT", "15m", self.day_ts(1), self.day_ts(2)
            )
        )

        self.verify_index_json("ETHUSDT", "15m", [[self.day_ts(0), self.day_ts(3)]])

    def test_download_chunked(self):
        self.setup_database()
        self.cachingInterface.chunk_size = 100

        chunk_length_ms = 60 * 15 * 1000

        list(
            self.cachingInterface.get_candles(
                "ETHUSDT",
                "15m",
                self.day_ts(0),
                self.day_ts(0) + 300 * chunk_length_ms,
            )
        )

        self.verify_index_json(
            "ETHUSDT",
            "15m",
            [
                [
                    self.day_ts(0) + 000 * chunk_length_ms,
                    self.day_ts(0) + 100 * chunk_length_ms,
                ],
                [
                    self.day_ts(0) + 100 * chunk_length_ms,
                    self.day_ts(0) + 200 * chunk_length_ms,
                ],
                [
                    self.day_ts(0) + 200 * chunk_length_ms,
                    self.day_ts(0) + 300 * chunk_length_ms,
                ],
            ],
        )

    def test_sequential(self):
        self.setup_database()
        self.cachingInterface.consolidate = True
        self.cachingInterface.chunk_size = 10

        for i in range(100):
            # i hours
            from_time = i * 1000 * 60 * 60 + self.day_ts(0)

            # i + 1 hours
            to_time = (i + 1) * 1000 * 60 * 60 + self.day_ts(0)

            data = list(
                self.cachingInterface.get_candles("ETHUSDT", "1h", from_time, to_time)
            )
            self.verify_data(data, "ETHUSD", "1h", from_time, to_time)

        valid_index_timestamps = [
            [
                self.day_ts(0) + i * 10 * 1000 * 60 * 60,
                self.day_ts(0) + (i + 1) * 10 * 1000 * 60 * 60,
            ]
            for i in range(10)
        ]

        self.verify_index_json("ETHUSDT", "1h", valid_index_timestamps)


if __name__ == "__main__":
    unittest.main()
