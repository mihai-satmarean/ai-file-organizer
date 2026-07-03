# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""Secret handling: masking, .env parsing/saving, key resolution. Nothing leaks."""

import os
import stat

from llm.secrets import mask, _parse_dotenv, resolve_api_key, save_key_to_dotenv


def test_mask_never_reveals_full_key():
    assert mask("sk-ant-abcdef1234567890") == "sk-…7890"
    assert mask("short") == "•••••"
    assert mask(None) == "<none>"
    assert "abcdef" not in mask("sk-ant-abcdef1234567890")


def test_parse_dotenv(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# comment\n"
        "export ANTHROPIC_API_KEY='sk-quoted'\n"
        "OPENAI_API_KEY=sk-plain\n"
        "\n"
        "BAD LINE NO EQUALS\n"
    )
    vals = _parse_dotenv(p)
    assert vals["ANTHROPIC_API_KEY"] == "sk-quoted"
    assert vals["OPENAI_API_KEY"] == "sk-plain"
    assert "BAD LINE NO EQUALS" not in vals


def test_resolve_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fromenv")
    res = resolve_api_key(["OPENAI_API_KEY"], allow_prompt=False, project_dir=tmp_path)
    assert res.value == "sk-fromenv"
    assert res.source == "env:OPENAI_API_KEY"


def test_resolve_missing_non_interactive(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = resolve_api_key(["OPENAI_API_KEY"], allow_prompt=False, project_dir=tmp_path)
    assert res.value is None
    assert res.source == "missing"
    assert not res  # falsy


def test_resolve_keyless(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    res = resolve_api_key(["OLLAMA_API_KEY"], keyless=True, allow_prompt=False, project_dir=tmp_path)
    assert res.value is None
    assert res.source == "keyless"


def test_save_key_to_dotenv_roundtrip_and_perms(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = save_key_to_dotenv("ANTHROPIC_API_KEY", "sk-secret-xyz", project_dir=tmp_path)
    assert path == tmp_path / ".env"

    # Owner-only permissions (0600).
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600

    # It's resolvable afterward and present in the process env.
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-secret-xyz"
    res = resolve_api_key(["ANTHROPIC_API_KEY"], allow_prompt=False, project_dir=tmp_path)
    assert res.value == "sk-secret-xyz"


def test_save_key_updates_existing_line(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=old\nOTHER=keep\n")
    save_key_to_dotenv("ANTHROPIC_API_KEY", "new", project_dir=tmp_path)
    text = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=new" in text
    assert "old" not in text
    assert "OTHER=keep" in text  # other entries preserved
