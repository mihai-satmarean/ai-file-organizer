"""Config never persists or accepts secrets, and ignores unknown keys."""

import json

from config import OrganizerConfig


def test_loading_drops_api_key(tmp_path, capsys):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({
        "backend": "llm",
        "llm_provider": "openai",
        "api_key": "sk-SHOULD-VANISH",
        "token": "also-secret",
    }))
    cfg = OrganizerConfig.from_json(p)
    assert not hasattr(cfg, "api_key")
    assert cfg.backend == "llm"
    assert cfg.llm_provider == "openai"
    # A warning is printed about the dropped secret.
    assert "api_key" in capsys.readouterr().out.lower()


def test_loading_ignores_unknown_keys(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"backend": "local", "totally_unknown": 123}))
    cfg = OrganizerConfig.from_json(p)
    assert cfg.backend == "local"
    assert not hasattr(cfg, "totally_unknown")


def test_saving_never_writes_secret_fields(tmp_path):
    cfg = OrganizerConfig(backend="llm", llm_provider="anthropic")
    # Even if someone monkeypatches a secret attribute, it's filtered on save.
    cfg.__dict__["api_key"] = "sk-LEAK"
    out = tmp_path / "saved.json"
    cfg.to_json(out)
    text = out.read_text()
    assert "sk-LEAK" not in text
    assert "api_key" not in text
    # Real settings survive the round trip.
    assert json.loads(text)["llm_provider"] == "anthropic"
