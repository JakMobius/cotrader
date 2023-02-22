#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig

from cotrader.trainer import Trainer


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    Trainer(cfg).run()
    return 0


if __name__ == "__main__":
    train()
