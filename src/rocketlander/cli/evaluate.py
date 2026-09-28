"""Evaluate an agent on the held-out seeds: rl-eval --agent runs/ppo_L0_s1/model.pt --level L0."""

from __future__ import annotations

from rocketlander.cli.common import agent_argument, base_parser, make_agent
from rocketlander.evaluation import EVAL_SEEDS, evaluate


def main(argv: list[str] | None = None) -> None:
    parser = base_parser("Evaluate an agent on held-out start states.", seed=False)
    agent_argument(parser)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    args = parser.parse_args(argv)
    summary = evaluate(make_agent(args.agent), args.level, EVAL_SEEDS[: args.episodes])
    print(
        f"{args.agent} on {args.level}: success {100 * summary.success_rate:.1f}% "
        f"over {summary.episodes} episodes"
    )
    print(
        f"  task return {summary.mean_task_return:.1f}   "
        f"fuel used {summary.mean_fuel_used:.0f} kg   "
        f"touchdown {summary.mean_touchdown_speed:.2f} m/s"
    )
    for reason, count in summary.reasons.most_common():
        print(f"  {reason:<24}{count}")


if __name__ == "__main__":
    main()
