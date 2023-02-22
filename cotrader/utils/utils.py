import argparse
from typing import Any, Tuple

import dateparser
import torch


def interval_to_ms(interval: str) -> int:
    if interval[-1] == "s":
        return int(interval[:-1]) * 1000
    elif interval[-1] == "m":
        return int(interval[:-1]) * 60 * 1000
    elif interval[-1] == "h":
        return int(interval[:-1]) * 60 * 60 * 1000
    elif interval[-1] == "d":
        return int(interval[:-1]) * 24 * 60 * 60 * 1000
    elif interval[-1] == "w":
        return int(interval[:-1]) * 7 * 24 * 60 * 60 * 1000
    elif interval[-1] == "M":
        return int(interval[:-1]) * 30 * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"{interval} is not a valid interval")


def binance_interval_argument(str):
    if interval_to_ms(str) is None:
        raise argparse.ArgumentTypeError(f"Invalid interval: {str}")
    return str


def normalize_timeframe(
    start_time: int, end_time: int, interval: str
) -> Tuple[int, int]:
    interval_ms = interval_to_ms(interval)
    start_time = start_time - start_time % interval_ms
    end_time = end_time - end_time % interval_ms
    return int(start_time), int(end_time)


def normalize_time_string(input: Any) -> str:
    parsed = dateparser.parse(str(input))
    assert parsed is not None, f"{input} is not a valid date"
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def softplus(input: torch.Tensor) -> torch.Tensor:
    return torch.where(input > 20, input, torch.log1p(torch.exp(input)))


def inv_softplus(input: torch.Tensor) -> torch.Tensor:
    return torch.where(input > 20, input, torch.log(torch.expm1(input)))
