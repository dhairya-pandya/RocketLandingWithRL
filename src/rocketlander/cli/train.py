"""Train an agent: rl-train --algo ppo --level L0 --seed 1."""

from __future__ import annotations

import time
from dataclasses import fields
from pathlib import Path

import torch
import yaml

from rocketlander.agents.registry import ALGORITHMS
from rocketlander.cli.common import base_parser
from rocketlander.common.checkpoint import load_checkpoint, save_checkpoint
from rocketlander.common.config import load_config
from rocketlander.common.curriculum import parse_level
from rocketlander.common.logger import Logger
from rocketlander.evaluation import EVAL_SEEDS, evaluate


def main(argv: list[str] | None = None) -> None:
    parser = base_parser("Train a learning agent on a level, mix or curriculum.", level_specs=True)
    parser.add_argument("--algo", default="ppo", choices=sorted(ALGORITHMS))
    parser.add_argument("--total-steps", type=int, default=None, help="override the YAML value")
    parser.add_argument("--init", type=Path, default=None, help="checkpoint to fine-tune from")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one YAML value, e.g. --set ent_coef=0.003 (repeatable)",
    )
    args = parser.parse_args(argv)

    torch.set_num_threads(1)  # small networks: one thread is fastest and leaves cores free
    try:  # validate the spec before anything is written
        eval_level, schedule = parse_level(args.level, args.seed)
    except ValueError as error:
        parser.error(str(error))
    algo = ALGORITHMS[args.algo]
    if schedule is not None and not algo.level_schedules:
        parser.error(f"{args.algo} trains on a single level, not {args.level!r}")
    overrides = {"total_steps": args.total_steps} if args.total_steps else {}
    known = {f.name for f in fields(algo.config_cls)}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or key not in known:
            parser.error(f"--set {item!r}: expected KEY=VALUE with KEY one of {sorted(known)}")
        overrides[key] = yaml.safe_load(value)  # typed like YAML: 0.003, 4, [128, 128], true
    config = load_config(algo.config_cls, args.algo, overrides)
    name = args.level.replace(":", "_").replace(",", "")
    run_dir = args.run_dir or Path("runs") / f"{args.algo}_{name}_s{args.seed}"
    init_agent = load_checkpoint(args.init)[0] if args.init else None
    if init_agent is not None:
        init_agent.train()
        init_agent.obs_rms.soften(max_count=1e4)  # let the stats adapt to the new level

    logger = Logger(run_dir, verbose=not args.quiet)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(vars(config), sort_keys=False))
    start = time.time()
    agent = algo.train(config, args.level, args.seed, logger, init_agent)
    logger.close()

    summary = evaluate(agent, eval_level, EVAL_SEEDS)
    save_checkpoint(
        run_dir / "model.pt",
        agent,
        args.algo,
        config,
        level=args.level,
        seed=args.seed,
        eval_success_rate=summary.success_rate,
        init=str(args.init) if args.init else None,
    )
    minutes = (time.time() - start) / 60
    print(
        f"Trained {args.algo} on {args.level} in {minutes:.1f} min. "
        f"Held-out success on {eval_level} {100 * summary.success_rate:.0f}% "
        f"over {summary.episodes} episodes. "
        f"Saved {run_dir / 'model.pt'}"
    )


if __name__ == "__main__":
    main()
