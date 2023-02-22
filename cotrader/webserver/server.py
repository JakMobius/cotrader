import time
from pathlib import Path

from sanic import Request, Sanic, Websocket
from sanic.response import json
from sanic.worker.loader import AppLoader
from sanic_ext import validate

from cotrader.utils.utils import interval_to_ms
from cotrader.webserver.session import CandlesSocketParams, GetCandlesRequest, Server


def create_app():
    web_root = Path("cotrader/web_app/dist")
    src_root = Path("cotrader/web_app/src")
    models_root = Path("models")

    app = Sanic("CoTrader")
    app.static("/", web_root, index="index.html", name="Dist")
    app.static("/src", src_root, name="Sources")

    server = Server()

    @app.get("/api/get-candles")
    @validate(query=GetCandlesRequest)
    async def get_candles(request: Request, query: GetCandlesRequest):
        try:
            interval_ms = interval_to_ms(query.interval)
        except ValueError:
            return json({"error": ""})
        if query.end < query.start:
            return json({"error": "end < start"}, status=400)
        if (query.end - query.start) > 1024 * interval_ms:
            return json({"error": "too much candles"}, status=400)

        start = query.start
        end = query.end

        current_time_ms = int(time.time() * 1000)
        end = min(end, current_time_ms)
        start = min(start, end)

        candles = server.datasource.get_candles(
            symbol=query.symbol,
            interval=query.interval,
            start_time=start,
            end_time=end,
        )

        return json({"result": [candle.serialize() for candle in candles]})

    @app.get("/api/get-predictors")
    async def get_predictors(request: Request):
        predictors = [
            {"id": "real", "name": "Real values", "real": True},
        ]

        # Recursively find all directories containing config.yaml
        for entry in models_root.rglob("config.yaml"):
            predictor_dir = entry.parent
            predictors.append({"id": str(predictor_dir.relative_to(models_root))})

        # Sort by modification time of config.yaml, newest first
        predictors[1:] = sorted(
            predictors[1:],
            key=lambda pred: (models_root / pred["id"] / "config.yaml").stat().st_mtime,
            reverse=True,
        )
        return json({"result": predictors})

    @app.websocket("/predict")
    async def predict(request: Request, ws: Websocket):
        await server.predict(ws)

    @app.websocket("/candles")
    @validate(query=CandlesSocketParams)
    async def candles(request: Request, ws: Websocket, query: CandlesSocketParams):
        await server.candles_socket(ws, query)

    return app


def run():
    loader = AppLoader(factory=create_app)
    app = loader.load()
    app.prepare(port=8000, dev=True)
    Sanic.serve(primary=app, app_loader=loader)
