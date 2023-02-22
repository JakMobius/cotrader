from typing import Optional

import torch
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger


def get_trainer_loggers(cfg, experiment_dir: Optional[str] = None):
    loggers = []

    if cfg.output.tensorboard.enabled is True:
        loggers.append(
            TensorBoardLogger(
                save_dir=(
                    cfg.output.tensorboard.dir
                    if experiment_dir is None
                    else experiment_dir
                ),
                name=cfg.output.name,
            )
        )

    if cfg.output.mlflow.enabled is True:
        loggers.append(
            MLFlowLogger(
                experiment_name=(
                    cfg.output.mlflow.experiment_name
                    if experiment_dir is None
                    else experiment_dir
                ),
                run_name=cfg.output.name,
                tracking_uri=cfg.output.mlflow.tracking_uri,
            )
        )

    return loggers


def get_loss_function(cfg):
    if cfg.training.loss_function == "mae":
        return torch.nn.L1Loss()
    if cfg.training.loss_function == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unknown loss function: ${cfg.training.loss_function}")
