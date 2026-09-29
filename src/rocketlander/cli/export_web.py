"""Export missions and agents' flights for the web game: rl-export-web --config web-export.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rocketlander.web_export import export, load_export_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export missions and flights for the web game.")
    parser.add_argument("--config", type=Path, default=Path("web-export.yaml"))
    args = parser.parse_args(argv)
    try:
        config = load_export_config(yaml.safe_load(args.config.read_text()))
    except ValueError as error:
        parser.error(str(error))
    for path in export(config):
        print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
