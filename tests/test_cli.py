"""Cost-safety invariants: LLM-calling commands refuse to run without
--confirm-spend and never construct a client before that check."""
import textwrap

import pytest

from avo.cli import main


@pytest.fixture
def config_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""\
        run_name: t
        task: tasks/sort_py
        llm:
          provider: anthropic
          model: claude-opus-5
    """))
    return p


def test_run_refuses_without_confirm_spend(config_file, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", str(config_file)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "--confirm-spend" in out and "claude-opus-5" in out
    assert "NOT SET" in out  # surfaces missing key without reading it


def test_resume_refuses_without_confirm_spend(tmp_path, config_file, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(config_file.read_text())
    with pytest.raises(SystemExit) as exc:
        main(["resume", "--run", str(run_dir)])
    assert exc.value.code == 1


def test_run_with_confirm_but_no_key_fails_before_network(config_file,
                                                          monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key"):
        main(["run", "--config", str(config_file), "--confirm-spend"])


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ["run", "resume", "eval-once", "baselines", "report", "rebench"]:
        assert cmd in out
