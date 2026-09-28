"""Evaluate agents on the held-out seeds: rl-eval --agent runs/ppo_L0_s1/model.pt --level L0.

Several --agent paths (e.g. one per training seed) also print their interquartile mean.
"""

from __future__ import annotations

import numpy as np

from rocketlander.cli.common import base_parser, make_agent
from rocketlander.evaluation import EVAL_SEEDS, EvalSummary, evaluate
from rocketlander.stats import bootstrap_ci, iqm


def report(name: str, summary: EvalSummary) -> None:
    low, high = summary.success_ci
    print(
        f"{name} on {summary.level}: success {100 * summary.success_rate:.1f}% "
        f"(95% CI {100 * low:.0f}-{100 * high:.0f}) over {summary.episodes} episodes"
    )
    print(
        f"  task return {summary.mean_task_return:.1f}   "
        f"fuel used {summary.mean_fuel_used:.0f} kg   "
        f"touchdown {summary.mean_touchdown_speed:.2f} m/s"
    )
    for reason, count in summary.reasons.most_common():
        print(f"  {reason:<24}{count}")


def main(argv: list[str] | None = None) -> None:
    parser = base_parser("Evaluate agents on held-out start states.", seed=False)
    parser.add_argument("--agent", nargs="+", default=["pid"], help="pid, random or .pt paths")
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    args = parser.parse_args(argv)
    rates = []
    for name in args.agent:
        summary = evaluate(make_agent(name), args.level, EVAL_SEEDS[: args.episodes])
        report(name, summary)
        rates.append(summary.success_rate)
    if len(rates) > 1:
        low, high = bootstrap_ci(np.array(rates), iqm)
        print(
            f"IQM success over {len(rates)} agents: {100 * iqm(np.array(rates)):.1f}% "
            f"(bootstrap 95% CI {100 * low:.0f}-{100 * high:.0f})"
        )


if __name__ == "__main__":
    main()
