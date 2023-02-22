import csv
import json
import os
from typing import Iterable, Optional

from tqdm import tqdm

from cotrader.datasource.candles.candle_datasource import Candle, CandleDatasource
from cotrader.utils.utils import interval_to_ms, normalize_timeframe


class CachingDatasource(CandleDatasource):
    database_dir: str
    interface: Optional[CandleDatasource]
    consolidate: bool
    chunk_size: int

    def __init__(
        self,
        interface: Optional[CandleDatasource] = None,
        database_dir: str = "dataset",
    ):
        self.database_dir = database_dir
        self.interface = interface
        self.consolidate = True
        self.chunk_size = 16384

    def get_candles(
        self, symbol, interval, start_time: int, end_time: int
    ) -> Iterable[Candle]:
        # Returns cached data if available, otherwise fetches data from binance
        # and saves it to the database

        data_controller = self.get_controller_for(symbol, interval)
        return data_controller.get_candles(start_time, end_time)

    def get_controller_for(self, symbol, interval):
        cache_dir = os.path.join(self.database_dir, symbol, interval)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        return DataController(self, cache_dir, symbol, interval)

    @classmethod
    def filename_for_timeframe(cls, time_from, time_to):
        return f"data-{time_from}-{time_to}.csv"

    @staticmethod
    def load_candles(file_path: str):
        candles = []
        with open(file_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                candle = Candle(int(row[0]))
                for i, k in enumerate(header):
                    if k != "timestamp":
                        candle.set(k, float(row[i]))

                candles.append(candle)
        return candles

    @staticmethod
    def save_candles(candles: list[Candle], file_path: str):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = list(candles[0]._indicators.keys())

            writer.writerow(["timestamp", *header])
            for candle in candles:
                values = [candle.get(key) for key in header]
                writer.writerow([candle.timestamp, *values])


class DataController:
    database: CachingDatasource
    directory: str
    symbol: str
    interval: str
    index_dirty: bool
    index: list[dict]

    def __init__(
        self,
        database: CachingDatasource,
        directory: str,
        symbol: str,
        interval: str,
    ):
        self.directory = directory
        self.database = database
        self.symbol = symbol
        self.interval = interval
        self.index_dirty = False

        # Read the index.json if it exists
        # index.json stores information about the data in the directory
        # It is a dictionary with the following structure:
        # [
        #   {
        #       "from": 1234567890,
        #       "to": 1234567890,
        #       "file": "data-1234567890-1234567890.csv"
        #   },
        #   ...
        # ]
        # Elements in the list are sorted by "from" in ascending order

        self.index_path = os.path.join(self.directory, "index.json")
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = []
            self.index_dirty = True

    def save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self.index, f)
        self.index_dirty = False

    def find_index(self, time: int):
        """
        Find the index of the first element in the `index` list where
        "from" <= time < "to". This method uses a binary search algorithm to
        find the desired index quickly.

        Args:
            time (int): The time value to search for.
            side (int, optional): Determines the comparison logic for the search.
                                  Defaults to 0.

        Returns:
            int or None: The index of the matching element in the `index` list,
                         or `None` if no such element is found.
        """

        left = 0
        right = len(self.index)

        while left < right:
            mid = (left + right) // 2

            if self.index[mid]["from"] <= time < self.index[mid]["to"]:
                return mid
            elif time < self.index[mid]["from"]:
                right = mid
            else:
                left = mid + 1

        return None

    def download_data(
        self, start_time: int, end_time: int, progress: Optional[tqdm] = None
    ) -> tuple[int, int]:
        if not self.database.interface:
            raise Exception("The requested data is not available in the database")

        assert start_time < end_time
        # Download data from binance from start_time to end_time by chunks of
        # database.chunk_size elements, then insert the downloaded data into
        # the index at insert_index, then save the index.json, and return the
        # number of downloaded chunks

        insert_index = self.find_nearest_left_index(start_time) + 1

        interval_ms = interval_to_ms(self.interval)

        download_interval = self.database.chunk_size * interval_ms

        chunks_downloaded = 0

        while start_time < end_time:
            end = min(start_time + download_interval, end_time)

            # Save data to file

            file_name = CachingDatasource.filename_for_timeframe(start_time, end)
            file_path = os.path.join(self.directory, file_name)

            max_candles = (end - start_time) // interval_ms

            candles: list[Candle] = []
            for i, candle in enumerate(
                self.database.interface.get_candles(
                    self.symbol, self.interval, start_time, end
                )
            ):
                if progress is not None:
                    progress.update(1)

                if i >= max_candles:
                    break
                candles.append(candle)

            # Make sure that binance returns the correct number of elements
            assert len(candles) == max_candles

            CachingDatasource.save_candles(candles, file_path)

            # Insert data into index

            self.index.insert(
                insert_index,
                {"from": start_time, "to": end, "file": file_name},
            )

            insert_index += 1
            chunks_downloaded += 1

            start_time = end

        if chunks_downloaded > 0:
            self.index_dirty = True

        return insert_index - chunks_downloaded, chunks_downloaded

    def flush_index(self):
        if self.index_dirty:
            self.save_index()

    def find_nearest_left_index(self, time: int):
        # input time should not be in the index
        # find the nearest boundary of the downloaded data
        # if specified time is before the first boundary, return -1

        if len(self.index) == 0:
            return -1

        # Search for the nearest boundary to the input time using binary search

        left = 0
        right = len(self.index)

        while left < right:
            mid = (left + right) // 2

            # input time should not be in the index
            assert not self.index[mid]["from"] <= time < self.index[mid]["to"]

            if time < self.index[mid]["from"]:
                right = mid
            else:
                left = mid + 1

        # We should get the index of the nearest boundary to the input time
        # from either the left or the right

        is_nearest_to_left = left in self.index and self.index[left]["to"] <= time

        if not is_nearest_to_left:
            left -= 1

        assert (left == -1) == (time < self.index[0]["from"])
        assert (left == len(self.index) - 1) == (time >= self.index[-1]["to"])

        if 0 <= left < len(self.index) - 1:
            assert self.index[left]["to"] <= time < self.index[left + 1]["from"]

        return left

    def find_nearest_time(self, from_time: int, to_time: int):
        # from_time and to_time are timestamps in milliseconds
        # from_time should not be in the index
        # returns the nearest time to to_time that is in the index

        if len(self.index) == 0:
            return to_time

        nearest_left_index = self.find_nearest_left_index(from_time)

        if to_time >= from_time:
            if nearest_left_index < len(self.index) - 1:
                return min(to_time, self.index[nearest_left_index + 1]["from"])
            else:
                return to_time
        else:
            if nearest_left_index >= 0:
                return max(to_time, self.index[nearest_left_index]["to"])
            else:
                return to_time

    def count(self, start_time, end_time, downloaded=False):
        interval_ms = interval_to_ms(self.interval)
        start_time = (start_time // interval_ms) * interval_ms
        end_time = ((end_time + interval_ms - 1) // interval_ms) * interval_ms

        if not downloaded:
            return (end_time - start_time) // interval_ms

        result = 0

        for time in range(start_time, end_time, interval_ms):
            if self.find_index(time) is not None:
                result += 1

        return result

    def populate_index(self, start_time, end_time, progress=None):
        if start_time == end_time:
            return

        if progress is not None:
            total = self.count(start_time, end_time)
            downloaded = self.count(start_time, end_time, downloaded=True)
            progress.total = total - downloaded

        start_index = self.find_index(start_time)

        if start_index is None:
            (start_index, chunks) = self.download_data(
                start_time, self.find_nearest_time(start_time, end_time), progress
            )

        end_index = self.find_index(end_time - 1)

        if end_index is None:
            (end_index, chunks) = self.download_data(
                self.find_nearest_time(end_time - 1, start_time), end_time, progress
            )

            end_index += chunks - 1

        # Iterate over the index from start_index to end_index and make sure that the
        # data is continuous If the data is not continuous, download the missing data
        # from binance

        i = start_index
        while i < end_index:
            assert self.index or self.index[i]["to"] <= self.index[i + 1]["from"]

            if self.index[i]["to"] != self.index[i + 1]["from"]:
                (insert_index, chunks) = self.download_data(
                    self.index[i]["to"], self.index[i + 1]["from"], progress
                )
                assert insert_index == i + 1
                end_index += chunks
                i += chunks
            i += 1

        # If database configured to consolidate data, do so now
        if self.database.consolidate:
            self.consolidate_index()

        # Save the index if it is dirty
        self.flush_index()

    def consolidate_index(self):
        # Iterate through the index and search for adjacent files that can be
        # consolidated If two adjacent files can be consolidated, delete the
        # second file and update the first file's to_time

        i = 0
        interval_ms = interval_to_ms(self.interval)

        while i < len(self.index) - 1:
            if self.index[i]["to"] < self.index[i + 1]["from"]:
                i += 1
                continue
            # Check whether these two files are small enough to be consolidated

            first_file_entries = (
                self.index[i]["to"] - self.index[i]["from"]
            ) // interval_ms
            second_file_entries = (
                self.index[i + 1]["to"] - self.index[i + 1]["from"]
            ) // interval_ms

            if first_file_entries + second_file_entries > self.database.chunk_size:
                i += 1
                continue

            # The two files are adjacent, consolidate them

            file_path = os.path.join(self.directory, self.index[i]["file"])
            data = CachingDatasource.load_candles(file_path)

            file_path = os.path.join(self.directory, self.index[i + 1]["file"])
            data += CachingDatasource.load_candles(file_path)

            file_name = CachingDatasource.filename_for_timeframe(
                self.index[i]["from"], self.index[i + 1]["to"]
            )
            file_path = os.path.join(self.directory, file_name)
            CachingDatasource.save_candles(data, file_path)

            # Delete old files
            os.remove(os.path.join(self.directory, self.index[i]["file"]))
            os.remove(os.path.join(self.directory, self.index[i + 1]["file"]))

            # Update the index

            self.index[i]["to"] = self.index[i + 1]["to"]
            self.index[i]["file"] = file_name

            del self.index[i + 1]

            self.index_dirty = True

    def get_candles(self, start_time: int, end_time: int) -> Iterable[Candle]:
        # from_time and to_time are timestamps in milliseconds
        # returns a list of lists of the form:
        #   [timestamp, open, high, low, close, volume]
        # where start_time <= timestamp < end_time (half-open interval)

        start_time, end_time = normalize_timeframe(start_time, end_time, self.interval)
        if start_time == end_time:
            return
        self.populate_index(start_time, end_time)

        start_index = self.find_index(start_time)
        end_index = self.find_index(end_time - 1)

        if start_index is None or end_index is None:
            raise IndexError("Start or end index not found in the index list.")

        # Read the data from the files and return it

        interval_in_ms = interval_to_ms(self.interval)

        for i in range(start_index, end_index + 1):
            # Read the data from the file

            file_path = os.path.join(self.directory, self.index[i]["file"])
            file_data = CachingDatasource.load_candles(file_path)

            # Find the start and end indices of the data that we need to read from the
            # file If the file contains data that is not in the interval [start_time,
            # end_time), then we need to trim the data

            file_start_time = self.index[i]["from"]
            file_end_time = self.index[i]["to"]

            file_start_index = 0
            file_end_index = len(file_data)

            if start_time > file_start_time:
                file_start_index = int((start_time - file_start_time) // interval_in_ms)

            if end_time < file_end_time:
                file_end_index = int((end_time - file_start_time) // interval_in_ms)

            # Append the data to the list of data that we will return

            for candle_index in range(file_start_index, file_end_index):
                yield file_data[candle_index]
