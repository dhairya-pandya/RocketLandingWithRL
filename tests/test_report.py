import pytest
import yaml

from rocketlander.cli import compare
from rocketlander.report import (
    build_report,
    cell,
    kept_lessons,
    line_style,
    load_report_config,
    plot_learning_curves,
    read_curve,
)


def test_report_config_rejects_unknown_agent_names():
    raw = {
        "out": "R.md",
        "figures": "f",
        "levels": ["L0"],
        "episodes": 1,
        "agents": [{"name": "PID", "agent": "pid"}],
        "robustness_agents": ["PPO"],
    }
    with pytest.raises(ValueError, match="PPO"):
        load_report_config(raw)
    with pytest.raises(ValueError):
        load_report_config({**raw, "robustness_agents": [], "colour": "red"})


def test_cell_shows_the_rate_and_its_interval():
    assert cell(1.0, 100) == "100% (96–100)"
    assert cell(0.0, 100) == "0% (0–4)"


def test_read_curve_turns_steps_into_minutes(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "step,key,value\n1000,train/steps_per_second,100\n1000,eval/success_rate,0.5\n"
    )
    steps, minutes, success = read_curve(tmp_path)
    assert steps == [1000] and minutes == pytest.approx([1000 / 100 / 60]) and success == [0.5]


def test_build_report_writes_tables_and_figures_and_keeps_the_lessons(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_report_config(
        yaml.safe_load("""
out: RESULTS.md
figures: fig
levels: [L0]
episodes: 3
workers: 1
agents:
  - {name: PID, agent: pid}
  - {name: Random, agent: random}
robustness_episodes: 1
robustness_agents: [PID]
""")
    )
    (tmp_path / "RESULTS.md").write_text("old table\n\n## Lessons learned\n\nKeep me.\n")
    out = build_report(config)
    text = out.read_text()
    assert "| PID | **100% (44–100)** |" in text and "| Random | 0% (0–56) |" in text
    assert text.endswith("## Lessons learned\n\nKeep me.\n") and "old table" not in text
    assert (tmp_path / "fig" / "robustness.png").exists()


def test_line_style_colours_by_algorithm_and_dashes_level_mixes():
    assert line_style("checkpoints/ppo_L2.pt") == {"color": "tab:orange", "linestyle": "-"}
    assert line_style("checkpoints/ppo_mix.pt")["linestyle"] == "--"
    assert line_style("checkpoints/ppo_curriculum.pt")["linestyle"] == ":"
    assert line_style("pid")["color"] == "tab:gray"
    assert line_style("checkpoints/dqn_L0.pt")["color"] == "tab:pink"


def test_compare_cli_builds_a_report_from_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r.yaml").write_text(
        "out: R.md\nfigures: f\nlevels: [L0]\nepisodes: 1\nworkers: 1\n"
        "agents: [{name: PID, agent: pid}]\nrobustness_episodes: 1\n"
    )
    compare.main(["--report", "r.yaml"])
    assert (tmp_path / "R.md").exists() and "Wrote R.md" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        {"agents": [{"name": "PPO", "agent": "checkpoints/missing.pt"}]},
        {"agents": [{"name": "PDI", "agent": "pdi"}]},
        {"levels": ["L9"]},
        {"agents": [{"name": "PID", "agent": "pid"}, {"name": "PID", "agent": "random"}]},
    ],
)
def test_report_config_rejects_mistakes_before_evaluating(change):
    raw = {
        "out": "R.md",
        "figures": "f",
        "levels": ["L0"],
        "episodes": 1,
        "agents": [{"name": "PID", "agent": "pid"}],
    }
    with pytest.raises(ValueError):
        load_report_config({**raw, **change})


def test_learning_curves_skip_agents_trained_outside_the_report_levels(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.csv").write_text(
        "step,key,value\n1000,train/steps_per_second,100\n1000,eval/success_rate,0.5\n"
    )
    config = load_report_config(
        {
            "out": "R.md",
            "figures": "f",
            "levels": ["L0"],
            "episodes": 1,
            "agents": [{"name": "PID", "agent": "pid", "run": str(run), "trained_on": "L4"}],
        }
    )
    assert plot_learning_curves(config, tmp_path / "curves.png") is False


@pytest.mark.parametrize("heading", ["## Lessons Learned", "## Lessons", "##  lessons learned"])
def test_lessons_survive_a_renamed_heading(heading, tmp_path):
    assert kept_lessons(f"# Results\n\n{heading}\n\nMine.\n", tmp_path / "R.md") == (
        f"{heading}\n\nMine.\n"
    )


def test_a_report_without_lessons_is_backed_up_before_it_is_replaced(tmp_path):
    out = tmp_path / "R.md"
    kept = kept_lessons("hand-written notes\n", out)
    assert "Lessons learned" in kept
    assert (tmp_path / "R.md.bak").read_text() == "hand-written notes\n"
