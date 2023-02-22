import subprocess
from abc import ABC
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from cotrader.datasource.tensor.candle_dataset import CandleDataset


class Model(pl.LightningModule, ABC):
    debug: bool = False
    loss_function = nn.MSELoss()

    def save(self, filepath):
        """Save model state and configuration"""
        raise NotImplementedError()

    def load(self, filepath):
        """Save model state and configuration"""
        raise NotImplementedError()

    def sanitize_nans(
        self, name: str, tensor: torch.Tensor, x: Optional[torch.Tensor] = None
    ):
        if not self.debug:
            return

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            for batch_idx in range(tensor.shape[0]):
                nan_indices = torch.nonzero(
                    torch.isnan(tensor[batch_idx]), as_tuple=False
                )
                inf_indices = torch.nonzero(
                    torch.isinf(tensor[batch_idx]), as_tuple=False
                )
                if len(nan_indices) == 0 and len(inf_indices) == 0:
                    continue

                if len(nan_indices) == 0:
                    values = "inf"
                    indices = inf_indices
                else:
                    values = "NaN"
                    indices = nan_indices

                if x is None:
                    raise ValueError(
                        f"{name} contains {values} values at batch {batch_idx}"
                        f" on indices: {indices.tolist()}."
                    )
                else:
                    inputs = x[batch_idx].flatten().tolist()
                    sorted_inputs = sorted(
                        enumerate(inputs), key=lambda x: abs(x[1]), reverse=True
                    )
                    largest_n = 5
                    largest_inputs = sorted_inputs[:largest_n]
                    largest_str = "\n".join(
                        f"- [seq={idx // x.shape[2]}, "
                        f"feature={idx % x.shape[2]}] = {val:.4f}"
                        for idx, val in largest_inputs
                    )
                    raise ValueError(
                        f"{name} contains {values} values at batch {batch_idx}"
                        f" on indices: {indices.tolist()}."
                        f" Largest {largest_n} inputs in the corresponding batch: \n"
                        f"{largest_str}"
                    )


class FixedSequenceModel(Model, ABC):
    sequence_length: int


def load_model(cfg, dataset: CandleDataset):
    return hydra.utils.instantiate(
        cfg,
        input_features=sum(f.get_feature_count() for f in dataset.input_features),
        output_features=sum(f.get_feature_count() for f in dataset.output_features),
    )


def get_commit_id():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_model(model, cfg, output):
    """Saves the model to a file."""
    # Create directory if it doesn't exist
    Path(output).mkdir(exist_ok=True, parents=True)

    # Save the model
    model.save(Path(output) / "model.pt")

    struct_mode = OmegaConf.is_struct(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg_with_commit = OmegaConf.merge(cfg, {"git_commit": get_commit_id()})
    OmegaConf.set_struct(cfg, struct_mode)
    (Path(output) / "config.yaml").write_text(OmegaConf.to_yaml(cfg_with_commit))
