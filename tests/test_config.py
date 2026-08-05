import textwrap

import pytest

from avo.config import LLMConfig, RunConfig, load_run_config


def test_load_minimal_yaml(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        run_name: t
        task: tasks/sort_py
        llm:
          provider: anthropic
          model: claude-opus-5
    """))
    cfg = load_run_config(cfg_file)
    assert cfg.run_name == "t"
    assert cfg.runner.kind == "local"
    assert cfg.budgets.max_versions == 10
    assert cfg.kb_dirs == ["knowledge_base"]


def test_api_key_from_env_only(monkeypatch):
    cfg = LLMConfig(provider="anthropic", model="m")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cfg.resolve_api_key() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert cfg.resolve_api_key() == "sk-test"


def test_custom_key_env(monkeypatch):
    cfg = LLMConfig(provider="openai_compat", model="m", api_key_env="DEEPSEEK_KEY")
    monkeypatch.setenv("DEEPSEEK_KEY", "abc")
    assert cfg.key_env_name() == "DEEPSEEK_KEY"
    assert cfg.resolve_api_key() == "abc"


def test_invalid_provider_rejected():
    with pytest.raises(Exception):
        RunConfig.model_validate({"run_name": "x", "task": "t",
                                  "llm": {"provider": "gemini", "model": "m"}})


def test_runner_identity_differs():
    from avo.config import RunnerConfig
    a = RunnerConfig(kind="ssh", host="asus")
    b = RunnerConfig(kind="ssh", host="h100")
    assert a.identity() != b.identity()
