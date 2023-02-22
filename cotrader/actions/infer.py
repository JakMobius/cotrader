import json

import dateparser
import hydra
from omegaconf import DictConfig

from cotrader.utils.utils import interval_to_ms
from cotrader.webserver.server import Server


@hydra.main(config_path="../../configs", config_name="infer", version_base=None)
def infer(cfg: DictConfig):
    date = dateparser.parse(cfg.date)
    if date is None:
        raise ValueError(f"Bad date: {cfg.date}")
    timestamp = date.timestamp() * 1000
    interval_ms = interval_to_ms(cfg.interval)
    timestamp = int((timestamp // interval_ms) * interval_ms)

    server = Server()

    predictions = server.run_model(
        model=cfg.model,
        interval=cfg.interval,
        start=timestamp,
        predictions=cfg.predictions,
        symbol=cfg.symbol,
        progress_bar=False,
    )

    features = ["open", "high", "low", "close", "volume"]

    for prediction in predictions:
        logged_dict = {}
        logged_dict["timestamp"] = prediction.timestamp
        for feature in features:
            feature_value = prediction.get(feature)
            logged_dict[feature] = feature_value.item()
        print(json.dumps(logged_dict))

    return 0


if __name__ == "__main__":
    infer()
