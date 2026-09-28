"""Stress-test an agent: rl-robust --agent checkpoints/ppo_L3.pt --level L3 --param wind_mean."""

from __future__ import annotations

from rocketlander.cli.common import agent_argument, base_parser, make_agent
from rocketlander.evaluation import EVAL_SEEDS
from rocketlander.robustness import sweep, sweep_values


def main(argv: list[str] | None = None) -> None:
    parser = base_parser(
        "Success rate as one physical parameter moves past its training range.", seed=False
    )
    agent_argument(parser)
    parser.add_argument(
        "--param",
        choices=sorted(sweep_values()),
        action="append",
        help="repeatable; default: every sweep in configs/robustness.yaml",
    )
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args(argv)
    agent = make_agent(args.agent)
    for param in args.param or sorted(sweep_values()):
        print(f"{args.agent} on {args.level}, sweeping {param}:")
        for value, summary in sweep(agent, args.level, param, EVAL_SEEDS[: args.episodes]):
            low, high = summary.success_ci
            print(
                f"  {value:>6g}   success {100 * summary.success_rate:5.1f}%  "
                f"(95% CI {100 * low:.0f}-{100 * high:.0f})"
            )


if __name__ == "__main__":
    main()
