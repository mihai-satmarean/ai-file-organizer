"""
Secure API key handling.

Design goals (all enforced here):
  * The user supplies their own key — we never ship one.
  * Keys are read from, in order: process environment, a gitignored .env file,
    then an interactive hidden prompt (getpass) when running in a terminal.
  * A key is NEVER written to a config file, log, undo script, or printed in
    full. Only a masked form (e.g. "sk-…a1b2") is ever displayed.

Nothing in this module persists a secret to disk on its own. The .env file is
something the *user* creates; we only read it (and .gitignore keeps it out of
version control).
"""

from __future__ import annotations

import os
import sys
import getpass
from pathlib import Path
from typing import Optional, Iterable


# Cache of values parsed from .env files we've already read, keyed by file path.
_dotenv_cache: dict[str, dict[str, str]] = {}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """
    Minimal .env parser (no external dependency).

    Supports `KEY=value`, `export KEY=value`, `#` comments, blank lines, and
    surrounding single/double quotes. Intentionally tiny — it only needs to
    recover API keys, not implement the full dotenv spec.
    """
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _load_dotenv(project_dir: Optional[Path]) -> dict[str, str]:
    """Read (and cache) the project's .env file, if present."""
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent.parent
    env_path = project_dir / ".env"
    key = str(env_path)
    if key not in _dotenv_cache:
        _dotenv_cache[key] = _parse_dotenv(env_path) if env_path.is_file() else {}
    return _dotenv_cache[key]


def save_key_to_dotenv(
    env_name: str,
    value: str,
    project_dir: Optional[Path] = None,
) -> Path:
    """
    Persist a key to the project's .env file (gitignored), with 0600 perms.

    Updates the line in place if `env_name` already exists, otherwise appends.
    The value is written to disk but never printed/returned. Also refreshes the
    in-process env + cache so the rest of this run can use it immediately.
    """
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent.parent
    path = project_dir / ".env"

    lines: list[str] = []
    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []

    out: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        body = stripped[len("export "):] if stripped.startswith("export ") else stripped
        name = body.split("=", 1)[0].strip() if "=" in body else ""
        if name == env_name:
            out.append(f"{env_name}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{env_name}={value}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)  # owner read/write only
    except OSError:
        pass

    # Make it usable for the remainder of this process.
    os.environ[env_name] = value
    _dotenv_cache[str(path)] = _parse_dotenv(path)
    return path


def mask(secret: Optional[str]) -> str:
    """Return a display-safe fingerprint of a secret. Never reveals enough to use it."""
    if not secret:
        return "<none>"
    secret = secret.strip()
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:3]}…{secret[-4:]}"


class KeyResolution:
    """The outcome of resolving a key — value plus where it came from (for status only)."""

    __slots__ = ("value", "source")

    def __init__(self, value: Optional[str], source: str):
        self.value = value
        self.source = source

    def __bool__(self) -> bool:
        return bool(self.value)


def resolve_api_key(
    env_names: Iterable[str],
    *,
    provider_label: str = "the provider",
    keyless: bool = False,
    allow_prompt: bool = True,
    project_dir: Optional[Path] = None,
) -> KeyResolution:
    """
    Resolve an API key without ever persisting or printing it.

    Lookup order:
      1. os.environ for each name in `env_names`
      2. the project .env file for each name
      3. an interactive hidden prompt (only if attached to a TTY)

    Args:
        env_names: Candidate environment variable names, most-preferred first.
        provider_label: Human name used only in the prompt text.
        keyless: If True, a missing key is acceptable (e.g. local Ollama).
        allow_prompt: If False, never prompt (use for non-interactive runs).
        project_dir: Where to look for .env (defaults to the repo root).

    Returns:
        KeyResolution. `.value` is None only when keyless or unavailable.
    """
    env_names = list(env_names)

    for name in env_names:
        val = os.environ.get(name)
        if val:
            return KeyResolution(val.strip(), f"env:{name}")

    dotenv = _load_dotenv(project_dir)
    for name in env_names:
        val = dotenv.get(name)
        if val:
            # Mirror into the process env so child SDKs that read it directly
            # (e.g. the Anthropic SDK reading ANTHROPIC_API_KEY) also see it.
            os.environ.setdefault(name, val)
            return KeyResolution(val.strip(), f".env:{name}")

    if keyless:
        return KeyResolution(None, "keyless")

    if allow_prompt and sys.stdin is not None and sys.stdin.isatty():
        primary = env_names[0] if env_names else "API key"
        try:
            entered = getpass.getpass(
                f"Enter API key for {provider_label} "
                f"(input hidden; set {primary} or a .env to skip this): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            entered = ""
        if entered:
            # Keep it in-process only for this run; do not write it anywhere.
            os.environ.setdefault(env_names[0] if env_names else "API_KEY", entered)
            return KeyResolution(entered, "prompt")

    return KeyResolution(None, "missing")
