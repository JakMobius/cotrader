from collections import deque

import torch

from cotrader.datasource.candles.candle_datasource import Candle, Multicandle
from cotrader.models.model import FixedSequenceModel
from cotrader.utils.utils import interval_to_ms
from cotrader.webserver.cost_predictor import CostPredictor


class FixedSequencePredictor(CostPredictor):
    model: FixedSequenceModel
    stack: deque[deque[Candle]]

    def __init__(self, cfg, model=None, model_path=None):
        super().__init__(cfg, model=model, model_path=model_path)
        self.stack = deque()

    def get_required_window(self):
        return (
            self.model.sequence_length
            + self.datasource.lookbehind
            + self.get_predictor_lookbehind()
        )

    def push_candle(self, candle: Candle):
        assert candle.timestamp % 1000 == 0
        self.window.append(candle)
        while len(self.window) > self.window_size:
            self.window.popleft()

    def get_prediction(self, interval: str) -> Candle:
        interval_ms = interval_to_ms(interval)
        end = self.window[-1].timestamp + interval_ms
        start = end - self.model.sequence_length * interval_ms

        assert self.window[-self.model.sequence_length].timestamp == start, (
            f"Expected window candle {self.window[-self.model.sequence_length]} "
            f"to have timestamp {start}"
        )

        candles = list(self.datasource.get_candles("", interval, start, end))

        assert candles[0] == self.window[-self.model.sequence_length]
        assert candles[-1].timestamp == self.window[-1].timestamp

        feature_vector = torch.zeros(
            [
                len(candles),
                sum([feature.get_feature_count() for feature in self.input_features]),
            ]
        )

        for i in range(len(candles)):

            def candle_getter(idx):
                return Multicandle(candle=candles[i + idx])

            feature_vector[i] = torch.cat(
                [feature.flatten(candle_getter)[0] for feature in self.input_features]
            )

        feature_vector = feature_vector.unsqueeze(0)

        self.model.eval()
        prediction = self.model.forward(feature_vector)

        def window_accessor(idx):
            return Multicandle(candle=self.window[idx + len(self.window)])

        return self.extract_single_prediction(window_accessor, prediction, interval)

    def push_state(self):
        self.stack.append(self.window.copy())

    def pop_state(self):
        self.window = self.stack.pop()
