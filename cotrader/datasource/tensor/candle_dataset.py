import math
from abc import ABC
from collections import deque
from typing import Callable, List, Optional

import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from cotrader.datasource.candles.candle_datasource import CandleDatasource, Multicandle
from cotrader.utils.utils import (
    interval_to_ms,
    inv_softplus,
    normalize_timeframe,
    softplus,
)


class CandleFeature(ABC):
    name: str
    target: Optional[str]
    shape: list[int]

    def __init__(
        self,
        config: DictConfig,
        shape: list[int],
    ):
        self.name = config["name"] if "name" in config else self.__class__.__name__
        self.target = config["target"] if "target" in config else None
        self.shape = shape
        self.debug = config.debug if "debug" in config else False

    def get_feature_count(self):
        return int(torch.prod(torch.tensor(self.shape)))

    def flatten(self, candle_getter: Callable[[int], Multicandle]):
        """
        Flattens the multidimentional feature values into a single-dimension array.
        Returns shape [candle_idx, feature_idx]
        """
        value = self.get_value(candle_getter)
        if len(value.shape) > 1:
            return torch.flatten(self.get_value(candle_getter), start_dim=1)
        return value.unsqueeze(-1)

    def get_lookbehind(self) -> int:
        """Return the maximum lookbehind required for this feature"""
        return 0

    def get_lookahead(self) -> int:
        """Return the maximum lookahead required for this feature"""
        return 0

    def get_value(self, candle_getter: Callable[[int], Multicandle]) -> torch.Tensor:
        """
        Returns the feature value
        Returns shape [candle_idx, ...self.shape]
        """

        raise NotImplementedError()


def parse_property_config(config):
    """Parse a property config in the format [name, offset]"""
    assert (isinstance(config, ListConfig) or isinstance(config, list)) and len(
        config
    ) == 2, f"Expected [name, offset] format, got {config}"
    name = config[0]
    offset = int(config[1])
    return name, offset


class ValueFeature(CandleFeature):
    """Simple feature that directly uses a value from a candle or its indicators"""

    def __init__(self, config: DictConfig):
        super().__init__(config, [1])
        value_config = config["value"]
        self.value_name, self.candle_offset = parse_property_config(value_config)

    def get_lookbehind(self) -> int:
        return -min(0, self.candle_offset)

    def get_lookahead(self) -> int:
        return max(0, self.candle_offset)

    def get_value(self, candle_getter: Callable[[int], Multicandle]) -> torch.Tensor:
        return candle_getter(self.candle_offset).get(self.value_name)


class Log10FourierFeature(CandleFeature):
    """Feature that represents a numerical value as a point on unit circle using
    log10 fourier transform"""

    def __init__(self, config: DictConfig):
        super().__init__(config, [2])
        value_config = config["value"]
        self.value_name, self.candle_offset = parse_property_config(value_config)

    def get_lookbehind(self) -> int:
        return -min(0, self.candle_offset)

    def get_lookahead(self) -> int:
        return max(0, self.candle_offset)

    def get_value(self, candle_getter: Callable[[int], Multicandle]) -> torch.Tensor:
        candles = candle_getter(self.candle_offset)

        # Get value using helper function
        values = candles.get(self.value_name)

        # Convert to point on unit circle using log10 and phi
        phi = torch.log10(values) * math.pi * 2
        x = torch.cos(phi)
        y = torch.sin(phi)
        return torch.stack([x, y], dim=1)


class SoftplusRatioFeature(CandleFeature):
    BIAS = inv_softplus(torch.tensor(1.0))

    def __init__(self, config):
        super().__init__(config, [1])
        self.num_name, self.num_offset = parse_property_config(config["num"])
        self.den_name, self.den_offset = parse_property_config(config["den"])
        self.factor = config["factor"]
        if self.factor is None:
            self.factor = 1

    def get_lookbehind(self) -> int:
        return -min(0, self.num_offset, self.den_offset)

    def get_lookahead(self) -> int:
        return max(0, self.num_offset, self.den_offset)

    def get_value(self, candle_getter: Callable[[int], Multicandle]) -> torch.Tensor:
        exception = None
        # Get candles for numerator and denominator
        num_candles = candle_getter(self.num_offset)
        den_candles = candle_getter(self.den_offset)

        # Extract values using helper function
        num = num_candles.get(self.num_name)
        den = den_candles.get(self.den_name)

        ratio = None

        try:
            ratio = inv_softplus(num / den) - SoftplusRatioFeature.BIAS
        except ValueError as e:
            exception = e
        finally:
            if exception or self.debug:
                c1 = candle_getter(0)[0]
                c2 = candle_getter(0)[-1]

                print(
                    f"[DivSoftplusFeature] setting {self.name} for candles "
                    f"at {c1.timestamp}...{c2.timestamp}"
                )
                print(
                    f"[DivSoftplusFeature] [{self.num_offset}].{self.num_name} / "
                    f"[{self.den_offset}].{self.den_name} = {num} / {den} = {num / den}"
                )
            if exception:
                raise exception

        assert ratio is not None

        sanitycheck = torch.abs(ratio) > 10
        if sanitycheck.any():
            index = int(sanitycheck.nonzero()[0].item())
            print(
                f"[DivSoftplusFeature] Warning: [{self.name}] is too large "
                f"({ratio[index]}). Inputs are (num={num[index]}, den={den[index]}). "
                f"Candles: num={den_candles[index].timestamp},"
                f"den={num_candles[index].timestamp}"
            )

        return ratio * self.factor


class InvSoftplusRatioFeature(CandleFeature):
    def __init__(self, config: DictConfig):
        super().__init__(config, [1])
        self.ratio_name, self.ratio_offset = parse_property_config(config["ratio"])
        self.base_name, self.base_offset = parse_property_config(config["den"])
        self.factor = config.factor if "factor" in config else 1

    def get_lookbehind(self) -> int:
        return -min(0, self.ratio_offset, self.base_offset)

    def get_lookahead(self) -> int:
        return max(0, self.ratio_offset, self.base_offset)

    def get_value(self, candle_getter: Callable[[int], Multicandle]):
        ratio = candle_getter(self.ratio_offset).get(self.ratio_name)
        base = candle_getter(self.base_offset).get(self.base_name)

        return softplus(ratio / self.factor + SoftplusRatioFeature.BIAS) * base


class LogRatioFeature(CandleFeature):
    def __init__(self, config):
        self.num_name, self.num_offset = parse_property_config(config["num"])
        self.den_name, self.den_offset = parse_property_config(config["den"])

        super().__init__(config, [1])

    def get_lookbehind(self) -> int:
        return -min(0, self.num_offset, self.den_offset)

    def get_lookahead(self) -> int:
        return max(0, self.num_offset, self.den_offset)

    def get_value(self, candle_getter: Callable[[int], Multicandle]) -> torch.Tensor:
        num_candles = candle_getter(self.num_offset)
        den_candles = candle_getter(self.den_offset)

        num = num_candles.get(self.num_name)
        den = den_candles.get(self.den_name)

        ratio = torch.log(num / den)

        assert ratio.shape[1] == 1

        sanitycheck = torch.abs(ratio) > 10
        if sanitycheck.any():
            index = int(sanitycheck.nonzero()[0, 0])
            print(
                f"[LogRatioFeature] Warning: [{self.name}] is too large "
                f"({ratio[index]}). Inputs are (num={num[index]}, den={den[index]}). "
                f"Candles: num={den_candles[index].timestamp},"
                f"den={num_candles[index].timestamp}"
            )

        return ratio


class InvLogRatioFeature(CandleFeature):
    def __init__(self, config: DictConfig):
        self.ratio_name, self.ratio_offset = parse_property_config(config["ratio"])
        self.base_name, self.base_offset = parse_property_config(config["den"])

        super().__init__(config, [1])

    def get_lookbehind(self) -> int:
        return -min(0, self.ratio_offset, self.base_offset)

    def get_lookahead(self) -> int:
        return max(0, self.ratio_offset, self.base_offset)

    def get_value(self, candle_getter: Callable[[int], Multicandle]):
        ratio_candles = candle_getter(self.ratio_offset)
        base_candles = candle_getter(self.base_offset)

        ratio = ratio_candles.get(self.ratio_name)
        base = base_candles.get(self.base_name)

        return torch.exp(ratio) * base


FEATURE_CONSTRUCTORS = {
    "value": ValueFeature,
    "log10fourier": Log10FourierFeature,
    "inv_softplus_ratio": InvSoftplusRatioFeature,
    "softplus_ratio": SoftplusRatioFeature,
    "inv_log_ratio": InvLogRatioFeature,
    "log_ratio": LogRatioFeature,
}


class WrappedCandleDatasource:
    def __init__(
        self,
        datasource: CandleDatasource,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
    ):
        self.datasource = datasource
        self.symbol = symbol
        self.interval = interval
        self.start_time, self.end_time = normalize_timeframe(
            start_time, end_time, interval
        )
        self.interval_ms = interval_to_ms(self.interval)

    def __len__(self):
        return (self.end_time - self.start_time) // self.interval_ms

    def __iter__(self):
        """
        Iterates the datasource with candles in the range [start_time, end_time]
        """
        yield from self.get_candles()

    def get_candles(self, lookahead: int = 0, lookbehind: int = 0):
        start_time = self.start_time - self.interval_ms * lookbehind
        end_time = self.end_time + self.interval_ms * lookahead

        yield from self.datasource.get_candles(
            self.symbol, self.interval, start_time, end_time
        )


class CandleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasources: List[WrappedCandleDatasource],
        sequence_length: int,
        input_features_config: List[DictConfig],
        output_features_config: List[DictConfig],
        progress: Optional[str] = None,
        candle_batch_size: int = 1024,
        normalize_input_mean=False,
        normalize_output_mean=False,
    ):
        self.datasource_bases: List[int] = []
        self.timestamp_vectors: List[torch.Tensor] = []
        self.input_features_vectors: List[torch.Tensor] = []
        self.output_features_vectors: List[torch.Tensor] = []

        self.input_features = CandleDataset.build_features(input_features_config)
        self.output_features = CandleDataset.build_features(output_features_config)

        self.candle_batch_size = candle_batch_size
        self.sequence_length = sequence_length
        self.lookbehind = 0
        self.lookahead = 0
        self.normalize_input_mean = normalize_input_mean
        self.normalize_output_mean = normalize_output_mean

        self.input_feature_count = sum(
            [feature.get_feature_count() for feature in self.input_features]
        )
        self.output_feature_count = sum(
            [feature.get_feature_count() for feature in self.output_features]
        )

        for feature in self.input_features + self.output_features:
            self.lookbehind = max(self.lookbehind, feature.get_lookbehind())
            self.lookahead = max(self.lookahead, feature.get_lookahead())

        self.lookbehind += candle_batch_size

        tqdm_progress = None
        if progress is not None:
            tqdm_progress = tqdm(
                total=sum([len(source) for source in datasources]), desc=progress
            )

        base_idx = 0
        for datasource in datasources:
            self.process_datasource(datasource, tqdm_progress)
            self.datasource_bases.append(base_idx)
            base_idx += len(datasource) - self.sequence_length
        self.datasource_bases.append(base_idx)

    def process_datasource(
        self, datasource: WrappedCandleDatasource, tqdm_progress: Optional[tqdm]
    ):
        (timestamp_vector, input_features_vector, output_features_vector) = (
            self.setup_feature_vectors(datasource)
        )

        from_idx = 0

        for x_batch, y_batch, timestamps in self.extract_batches_from_datasource(
            datasource
        ):
            to_idx = from_idx + x_batch.shape[0]
            input_features_vector[from_idx:to_idx] = x_batch
            output_features_vector[from_idx:to_idx] = y_batch
            timestamp_vector[from_idx:to_idx] = timestamps

            if tqdm_progress:
                tqdm_progress.update(to_idx - from_idx)

            from_idx = to_idx

        assert from_idx == len(
            datasource
        ), f"Filled only {from_idx} / {len(datasource)} rows"
        assert timestamp_vector[0] == datasource.start_time
        assert timestamp_vector[-1] == datasource.end_time - datasource.interval_ms

        if self.normalize_input_mean:
            mean = input_features_vector.mean(dim=0, keepdim=True)
            input_features_vector = input_features_vector - mean

        if self.normalize_output_mean:
            mean = self.output_features_vector.mean(dim=0, keepdim=True)
            self.output_features_vector = self.output_features_vector - mean

        self.input_features_vectors.append(input_features_vector)
        self.output_features_vectors.append(output_features_vector)
        self.timestamp_vectors.append(timestamp_vector)

    def setup_feature_vectors(self, datasource: WrappedCandleDatasource):
        timestamp_vector = torch.zeros([len(datasource)], dtype=torch.long)
        input_features_vector = torch.zeros(
            [len(datasource), self.input_feature_count],
            dtype=torch.float32,
        )
        output_features_vector = torch.zeros(
            [len(datasource), self.output_feature_count],
            dtype=torch.float32,
        )

        return (timestamp_vector, input_features_vector, output_features_vector)

    def iter_windows(self, datasource: WrappedCandleDatasource):
        """
        Iterates the given WrappedCandleDatasource with candle windows of size
        lookbehind + lookahead + 1. Yields (candle_idx, window)
        """
        candles = datasource.get_candles(self.lookahead, self.lookbehind)
        iterator = enumerate(candles)
        window = deque()
        for i, candle in iterator:
            window.append(candle)
            if i < self.lookbehind:
                continue
            if len(window) > self.lookbehind + self.lookahead + 1:
                window.popleft()
            yield i - self.lookbehind, window

    def iter_batches(self, datasource: WrappedCandleDatasource):
        """
        Iterates the given WrappedCandleDatasource with batches windows of size
        candle_batch_size. Yields (window, batch), where window is from
        iter_windows
        """
        batch = list()
        for i, window in self.iter_windows(datasource):
            batch.append(window[-self.lookahead])
            if len(batch) >= self.candle_batch_size or i == len(datasource) - 1:
                yield window, batch
                batch.clear()

    def extract_batches_from_datasource(self, datasource: WrappedCandleDatasource):
        for window, batch in self.iter_batches(datasource):
            candles_in_batch = len(batch)

            getter_cache = dict()

            def candle_getter(candle_idx):
                if candle_idx in getter_cache:
                    return getter_cache[candle_idx]

                candles = list()
                for j in range(len(batch)):
                    candles.append(
                        window[
                            len(window)
                            + candle_idx
                            + j
                            - candles_in_batch
                            - self.lookahead
                        ],
                    )

                result = Multicandle(candles=candles)
                getter_cache[candle_idx] = result
                return result

            yield (
                torch.cat(
                    [feature.flatten(candle_getter) for feature in self.input_features],
                    dim=1,
                ),
                torch.cat(
                    [
                        feature.flatten(candle_getter)
                        for feature in self.output_features
                    ],
                    dim=1,
                ),
                candle_getter(0).timestamp,
            )

    def get_datasource_index(self, global_idx: int):
        left, right = 0, len(self.datasource_bases) - 1

        while left <= right:
            mid = (left + right) // 2
            left_boundary = self.datasource_bases[mid]
            right_boundary = self.datasource_bases[mid + 1]

            if left_boundary <= global_idx < right_boundary:
                return mid
            elif global_idx < left_boundary:
                right = mid - 1
            else:
                left = mid + 1

        assert False

    def __getitem__(self, index):
        datasource_idx = self.get_datasource_index(index)

        assert (
            self.datasource_bases[datasource_idx]
            <= index
            < self.datasource_bases[datasource_idx + 1]
        )

        local_idx = index - self.datasource_bases[datasource_idx]
        inputs = self.input_features_vectors[datasource_idx]
        outputs = self.output_features_vectors[datasource_idx]
        timestamps = self.timestamp_vectors[datasource_idx]

        input_seq = inputs[local_idx : local_idx + self.sequence_length]
        output_seq = outputs[local_idx + 1 : local_idx + self.sequence_length + 1]

        # Cost predictor needs the sequential candle index as if the
        # dataset wasn't trimmed to account for the sequence lengths.
        raw_index = index + datasource_idx * self.sequence_length

        return (
            input_seq,
            output_seq,
            raw_index,
            timestamps[local_idx],
        )

    def __len__(self):
        return self.datasource_bases[-1]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def print_statistics(self):
        print(f"Input features count: {self.input_features_vectors[0].shape[1]}")
        print(f"Output features count: {self.output_features_vectors[0].shape[1]}")
        print(f"Candle count: {len(self)}")

        def print_stats(features: List[CandleFeature], features_vectors, label):
            print(f"\n{label} features statistics:")
            start = 0

            vector = torch.cat(features_vectors, dim=0)

            for feature in features:
                count = feature.get_feature_count()
                data = vector[:, start : start + count]
                mean = data.mean(dim=0)
                std = data.std(dim=0)
                min_val = data.min(dim=0).values
                max_val = data.max(dim=0).values

                for i in range(count):
                    description = f"  {feature.name}"
                    if count > 1:
                        description += f"[{i}]"
                    description += ": "
                    description += f"mean={mean[i]:.4f}, "
                    description += f"std={std[i]:.4f}, "
                    description += f"min={min_val[i]:.4f}, "
                    description += f"max={max_val[i]:.4f}, "
                    print(description)
                start += count

        print_stats(self.input_features, self.input_features_vectors, "Input")
        print_stats(self.output_features, self.output_features_vectors, "Output")

    @classmethod
    def build_feature(cls, feature_config: DictConfig) -> CandleFeature:
        feature_type = feature_config.get("type")
        target_name = feature_config.get("target", feature_type)

        # Check if we have a constructor for this feature type
        assert feature_type in FEATURE_CONSTRUCTORS, f"Unkonwn feature {feature_type}"
        try:
            # Create feature with the appropriate constructor
            feature_class = FEATURE_CONSTRUCTORS[feature_type]
            return feature_class(feature_config)
        except (KeyError, AssertionError) as e:
            ft_type = str(feature_type)
            target = str(target_name)
            err_msg = f"Could not create feature {target} of type {ft_type}"
            print(f"{err_msg}: {str(e)}")
            raise

    @classmethod
    def build_features(cls, feature_configs: List[DictConfig]) -> List[CandleFeature]:
        features = []

        for feature_config in feature_configs:
            assert isinstance(feature_config, DictConfig)
            features.append(cls.build_feature(feature_config))

        return features
