"""Algorithm hyperparameters: dataclasses filled from rocketlander/configs/algos/<algo>.yaml."""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import resources
from typing import Any

import yaml


def load_config[T](config_cls: type[T], algo: str, overrides: dict[str, Any] | None = None) -> T:
    path = resources.files("rocketlander") / "configs" / "algos" / f"{algo}.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    config = config_from_dict(config_cls, raw)
    return replace(config, **(overrides or {}))


def config_from_dict[T](config_cls: type[T], raw: dict[str, Any]) -> T:
    known = {f.name for f in fields(config_cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown {config_cls.__name__} fields: {sorted(unknown)}")
    return config_cls(**raw)
