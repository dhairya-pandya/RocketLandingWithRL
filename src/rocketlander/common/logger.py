"""Writes training metrics to TensorBoard and to a long-format CSV (step, key, value)."""

from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_dir: Path, verbose: bool = True) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(run_dir))
        self.csv_file = (run_dir / "metrics.csv").open("w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["step", "key", "value"])
        self.verbose = verbose

    def log(self, step: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
            self.csv.writerow([step, key, value])
        self.csv_file.flush()
        if self.verbose:
            shown = "  ".join(f"{k} {v:.3g}" for k, v in metrics.items())
            print(f"[{step:>9}] {shown}", flush=True)

    def close(self) -> None:
        self.writer.close()
        self.csv_file.close()
