"""Save and load trained agents together with their config and provenance."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_checkpoint(path: Path, agent: Any, algo: str, config: Any, **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": algo,
            "architecture": agent.architecture(),
            "state_dict": agent.state_dict(),
            "extra_state": agent.extra_state(),
            "config": asdict(config),
            "git_commit": git_commit(),
            **meta,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[Any, dict[str, Any]]:
    """The agent (ready to act) and the checkpoint's metadata."""
    from rocketlander.agents.registry import ALGORITHMS  # late import: registry imports agents

    data = torch.load(path, weights_only=True)  # refuses pickled code from untrusted files
    agent = ALGORITHMS[data["algo"]].agent_cls(**data["architecture"])
    agent.load_state_dict(data["state_dict"])
    agent.load_extra_state(data["extra_state"])
    agent.eval()
    meta = {k: v for k, v in data.items() if k not in ("state_dict", "extra_state")}
    return agent, meta
