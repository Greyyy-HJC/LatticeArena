"""NN-tuned fit configuration for the gsfit_2pt task."""

from __future__ import annotations

import json
from pathlib import Path

from tasks.gsfit_2pt.interface import (
    GroundStateFitConfig,
    Pion2PtGroundStateFit,
    SubmissionMeta,
    config_from_dict,
)


CONFIG_PATH = Path(__file__).with_name("nn_config.json")


class NNTunedGroundStateFit(Pion2PtGroundStateFit):
    """Fit configuration produced by the tiny-MLP surrogate optimizer."""

    @property
    def meta(self) -> SubmissionMeta:
        payload = json.loads(CONFIG_PATH.read_text())
        meta = payload["meta"]
        return SubmissionMeta(
            name=str(meta["name"]),
            description=str(meta["description"]),
            authors=[str(author) for author in meta["authors"]],
        )

    @property
    def config(self) -> GroundStateFitConfig:
        payload = json.loads(CONFIG_PATH.read_text())
        return config_from_dict(payload["config"])
