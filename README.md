# RocketLandingWithRL

A 2D rocket learns to land on a moving, rocking drone ship in gusty wind. The physics, the
environment and every reinforcement learning algorithm are written from scratch in Python, trained
on a laptop CPU, and compared against a hand-written PID controller.

![A PPO agent landing on the rolling drone ship (level L2)](media/ppo_landing_L2.gif)

Status: **Phase 3 done**: environment, viewer, baselines, and the first learning agents (REINFORCE
and PPO) trained from scratch on a laptop CPU in about 6 minutes each. SAC, TD3, GRPO and CEM/ES come
next.

## Results so far

Success rate on 100 held-out start states per level (seeds 10000–10099, never used in training):

| Agent | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| Random | 0% | 0% | 0% | 0% | 0% |
| PID (hand-written) | 100% | 100% | 98% | 45% | 45% |
| REINFORCE, trained on L0 | 100% | 87% | 0% | 0% | 0% |
| PPO, trained on L0 | 97% | 52% | 0% | 0% | 0% |
| PPO, trained on L1 | 42% | 97% | 0% | 0% | 0% |
| PPO, trained on L2 | 46% | 85% | 94% | 55% | 10% |

Each agent is one seed, trained for 5 M steps (about 6 minutes); the checkpoints are in
`checkpoints/`. On its own level each PPO agent lands 94–97% of the time, and more softly than the
PID (0.97 m/s against 1.43 m/s on L2). The L2 agent even beats the PID on the unseen, windy L3.
Transfer is uneven, though: agents trained without a rolling deck never saw those observations
change and fail on L2+, and the L2 agent has unlearned the static pad of L0. Training across levels
(curriculum and domain randomization) is Phase 6. REINFORCE also masters L0, but only after about
4.5 M steps, against about 1 M for PPO.

## Train your own

```bash
uv run rl-train --algo ppo --level L2 --seed 1          # ~6 min on a laptop CPU
uv run rl-eval --agent runs/ppo_L2_s1/model.pt --level L2
uv run rl-watch --agent runs/ppo_L2_s1/model.pt --level L2
tensorboard --logdir runs                               # learning curves
```

Hyperparameters live in `src/rocketlander/configs/algos/ppo.yaml` and `reinforce.yaml`. `rl-train`
logs to TensorBoard and `runs/<name>/metrics.csv`, evaluates on held-out seeds during training,
and saves a checkpoint with its config and git commit.

## Quickstart

```bash
git clone https://github.com/dhairya-pandya/RocketLandingWithRL.git
cd RocketLandingWithRL
uv sync
uv run pytest
```

```python
from rocketlander.agents.pid import PIDAgent
from rocketlander.envs.rocket_env import RocketLanderEnv

env = RocketLanderEnv(level="L2")
agent = PIDAgent()
obs, _ = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(agent.act(obs))
    done = terminated or truncated
print(info["outcome"], info["reason"])
```

## Watch it, fly it, record it

```bash
uv run rl-watch --agent pid --level L2      # pid, random, or a checkpoint such as checkpoints/ppo_L2.pt
uv run rl-play --level L0                   # fly it yourself
uv run rl-record --agent pid --level L2 --seed 3 --out videos/landing.gif
```

| Keys | `rl-play` | `rl-watch` |
|---|---|---|
| W / S | throttle up / down (sticky) | |
| A / D | lean left / right (engine gimbal) | |
| Q / E | lean left / right (side thrusters, RCS) | |
| R / N | restart / new start position | N: next episode |
| Space | | pause |
| F V T P H | toggle force arrows, velocity, trail, plots, HUD | same |
| Esc | quit | quit |

The HUD turns each landing limit green once it is met (vertical and sideways speed, tilt, spin).
The mini-plots show the agent's actions, its reward per step and, for agents with a critic, its
value estimate V(s). Any script can also render through Gymnasium:
`RocketLanderEnv(level="L2", render_mode="human")` or `render_mode="rgb_array"`.

## The environment

**Physics.** The rocket is a 2D rigid body (`x, y, vx, vy, θ, ω`, fuel) integrated with
semi-implicit Euler at 60 Hz. The main engine's thrust is tilted by a ±15° gimbal; because it pushes
at the base, tilting the nozzle spins the rocket (`τ = −T·sin δ·L/2`). Side thrusters (RCS) add
rotation control. Quadratic drag uses air-relative velocity, which is how wind pushes the rocket.
Burning fuel makes the rocket lighter.

**The drone ship** sways, drifts, heaves and rolls, all as closed-form functions of time.
**Wind** is an Ornstein–Uhlenbeck process: a mean speed plus gusts that decay back to it.

**Touchdown.** The episode ends when a leg touches the deck or the sea. It is a landing only if
both legs are on the deck and, relative to the deck, the vertical speed is ≤ 2 m/s, the sideways
speed ≤ 1.5 m/s, the tilt ≤ 10° and the spin ≤ 0.3 rad/s. Otherwise the episode reports the reason
it crashed.

| | |
|---|---|
| Action | `Box(-1, 1, (3,))`: throttle, gimbal, RCS, at 30 Hz |
| Observation | 11 floats: position and velocity relative to the deck, `sin θ`, `cos θ`, `ω`, deck roll and roll rate, fuel remaining (tonnes), actual throttle. Wind is **not** observed. |
| Reward | Progress shaping `Φ(s') − Φ(s)` (closer, slower, more upright is better; `Φ` is kept at touchdown), fuel and time costs, +100 plus a softness bonus for landing, a deck crash graded by impact speed (−20 − 8·v, down to −100), and −100 for missing the ship, leaving the flight box or running out of fuel. `reward_mode="sparse"` keeps only the terminal reward. |
| Task reward | `info["task_reward"]` is the unshaped reward (terminal reward minus fuel and time costs). Evaluations and comparisons use it. |
| Snapshots | `get_state()` / `set_state()` / `reset(options={"state": s})` restore the env exactly, including the random number generator. |
| Checkpoints | `torch.load(..., weights_only=True)`: loading a checkpoint never runs pickled code. Training seeds are refused if they would reach the evaluation seeds. |

## Levels

| Level | What changes | Fuel | Time limit |
|---|---|---|---|
| L0 | Static pad, no wind, start close above | 1.0–1.2 t | 40 s |
| L1 | Ship sways and drifts | 1.3–1.5 t | 60 s |
| L2 | Ship also rolls and heaves | 1.5–1.7 t | 60 s |
| L3 | Wind gusts, randomized mass, thrust and engine lag | 1.5–1.7 t | 60 s |
| L4 | Booster return: far start at high speed | 2.8–3.2 t | 100 s |

Levels are YAML files in `src/rocketlander/configs/levels/` (angles in degrees, `_deg` keys);
add a file to add a level. The PID controller is tuned for nominal physics, so L3's randomized
rocket and wind are where it struggles: that gap is what the learning agents have to close.

## Why the reward looks the way it does

The first PPO runs did not land; they learned to **hover** until the time limit. Three changes fixed
it, each a textbook lesson:

1. **Discount.** With γ = 0.99 the agent looks about 3 s ahead, but landings are 10–30 s away.
   PPO uses γ = 0.999.
2. **Shaping that grades the touchdown.** Classic potential-based shaping (`Φ(terminal) = 0`)
   provably keeps the optimal policy, but it scores a crash at 3 m/s the same as one at 30 m/s, and
   with γ < 1 it pays a small bonus for every step spent alive. Keeping `Φ` at touchdown and grading
   deck crashes by impact speed gives the learner a slope toward "slower is better".
3. **Limited fuel.** With a 5 t tank the rocket could hover past the time limit, and a time-limit
   ending carries no penalty. Every level's tank now runs dry before its time limit, so hovering
   ends in an observable "out of fuel" failure. Closing this on L1–L2 as well raised PPO from
   89% to 97% on L1 and from 85% to 94% on L2.

## Development

```bash
uv run pytest                 # fast tests
uv run pytest -m slow         # learning tests (later phases)
uv run ruff check . && uv run ruff format --check .
```

## License

MIT
