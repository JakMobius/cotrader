import asyncio
import json
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ValidationError
from sanic import Websocket
from tqdm import tqdm

from cotrader.datasource.candles.binance_datasource import (
    BinanceDatasource,
    BinanceStream,
    BinanceTooEarlyError,
)
from cotrader.datasource.candles.caching_datasource import CachingDatasource
from cotrader.datasource.candles.candle_datasource import Candle, CandleDatasource
from cotrader.datasource.candles.dummy_datasource import DummyStream
from cotrader.utils.utils import interval_to_ms
from cotrader.webserver.fixed_sequence_predictor import FixedSequencePredictor


class RunModelRequest(BaseModel):
    model: str
    symbol: str
    interval: str
    start: int
    predictions: int


class GetCandlesRequest(BaseModel):
    symbol: str
    interval: str
    start: int
    end: int


class CandlesSocketParams(BaseModel):
    symbol: str
    interval: str


class RequestError(ValueError):
    def __init__(self, message="Invalid request"):
        super().__init__(message)


class Server:
    datasoure: CandleDatasource

    def __init__(self):
        binance_datasource = BinanceDatasource()
        self.datasource = CachingDatasource(binance_datasource)

    def predict_real(self, symbol: str, interval: str, start: int, predictions: int):
        interval_ms = interval_to_ms(interval)
        window_start = start
        window_end = start + predictions * interval_ms

        current_time_ms = int(time.time() * 1000)
        window_end = (min(window_end, current_time_ms) // interval_ms) * interval_ms

        try:
            yield from self.datasource.get_candles(
                symbol, interval, window_start, window_end
            )
        except BinanceTooEarlyError:
            raise RequestError("Can't predict from the future")

    async def predict(self, ws: Websocket):
        running = True

        while running:
            msg = await ws.recv()
            try:
                if msg is None:
                    raise ValueError("Received empty message from websocket")
                data = json.loads(msg)
                req = RunModelRequest(**data)
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                await ws.send(json.dumps({"type": "error", "message": str(e)}))
                running = False
                return

            try:
                for prediction in self.run_model(
                    req.model,
                    req.symbol,
                    req.interval,
                    req.start,
                    req.predictions,
                ):
                    await ws.send(
                        json.dumps(
                            {"type": "prediction", "candle": prediction.serialize()}
                        )
                    )
                    if not await self.wait_ack(ws):
                        break

            except (ValueError, AssertionError, RequestError) as e:
                raise
                await ws.send(json.dumps({"type": "error", "error": str(e)}))

            await ws.send(json.dumps({"type": "finish"}))

    async def wait_ack(self, ws: Websocket):
        ack_msg = await ws.recv()
        try:
            if not ack_msg:
                return
            ack_data = json.loads(ack_msg)
            return ack_data.get("type") == "ack"
        except Exception:
            return False

    def run_model(
        self,
        model: str,
        symbol: str,
        interval: str,
        start: int,
        predictions: int,
        progress_bar: bool = True,
    ):
        if model == "real":
            yield from self.predict_real(symbol, interval, start, predictions)

        interval_ms = interval_to_ms(interval)

        model_path = Path("models") / model
        cfg = OmegaConf.load(model_path / "config.yaml")
        predictor = FixedSequencePredictor(cfg, model_path=model_path / "model.pt")

        current_time_ms = int(time.time() * 1000)
        if current_time_ms + interval_ms < start:
            raise RequestError("Can't predict from the future")

        window_start = start - predictor.get_required_window() * interval_ms
        window_end = start
        candles = self.datasource.get_candles(
            symbol, interval, window_start, window_end
        )

        for candle in candles:
            predictor.push_candle(candle)

        prediction_range = range(predictions)
        if progress_bar:
            prediction_range = tqdm(prediction_range, unit="candle", desc="Predicting")

        for i in prediction_range:
            prediction = predictor.get_prediction(interval)
            predictor.push_candle(prediction)
            self.validate_candle(prediction)
            yield prediction

    def validate_candle(self, candle: Candle):
        for key, value in candle._indicators.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            else:
                assert math.isfinite(value)

    async def candles_socket(self, ws: Websocket, query: CandlesSocketParams):
        if query.symbol == "DUMMY":
            stream = DummyStream()
        else:
            stream = BinanceStream()

        await stream.subscribe(query.symbol, query.interval)

        loop = asyncio.get_event_loop()
        ws_task = loop.create_task(ws.recv())

        while True:
            candle_task = loop.create_task(stream.wait_for_candle())
            done, pending = await asyncio.wait(
                [ws_task, candle_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for coro in done:
                result = coro.result()
                if isinstance(result, Candle):
                    # Check that candle does not contain any NaNs or infs,
                    # so client-side JSON.parse does not get confused.

                    self.validate_candle(result)

                    await ws.send(
                        json.dumps(
                            {"type": "candle", "candle": result.serialize()},
                            allow_nan=False,
                        )
                    )
                else:
                    await stream.destroy()
                    print("Killing the binance socket")
                    return
