#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig

from cotrader.validator import Validator


@hydra.main(config_path="../../configs", config_name="validation", version_base=None)
def validate(cfg: DictConfig):
    Validator(cfg).run()
    return 0


if __name__ == "__main__":
    validate()
