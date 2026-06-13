"""
Interactive command-line wizard for the file organizer.

Walks the user through backend → provider → API key → model → folders →
action, then runs a preview or the organization. Run it with:

    python main.py            # (no args at a terminal launches this)
    python main.py -i
    python main.py --interactive

API-key hygiene is preserved: keys are entered hidden (getpass), shown only
masked, and saved only to the gitignored .env if the user opts in.
"""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Optional

from config import OrganizerConfig
from llm import PRESETS, list_presets, mask, resolve_api_key, save_key_to_dotenv


PROJECT_DIR = Path(__file__).resolve().parent

# Local-backend embedding model choices (name, blurb).
_EMBED_MODELS = [
    ("all-MiniLM-L6-v2", "fastest, ~90MB"),
    ("all-MiniLM-L12-v2", "balanced, ~120MB"),
    ("all-mpnet-base-v2", "most accurate, ~420MB"),
]


# ----------------------------- prompt helpers -----------------------------

def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        raise SystemExit(0)
    return answer or (default or "")


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        answer = _ask(f"{prompt} ({hint})").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer y or n.")


def _choose(title: str, options: list[tuple[str, str]], default_index: int = 0) -> str:
    """Render a numbered menu of (value, label) and return the chosen value."""
    print(f"\n{title}")
    for i, (_, label) in enumerate(options, 1):
        marker = " (default)" if i - 1 == default_index else ""
        print(f"  {i}. {label}{marker}")
    while True:
        raw = _ask("Choose", str(default_index + 1))
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        # Allow typing the value directly, too.
        for value, _ in options:
            if raw.lower() == value.lower():
                return value
        print(f"  Enter a number 1-{len(options)}.")


# ----------------------------- key + model --------------------------------

def _setup_key(preset) -> None:
    """
    Ensure a usable key exists for the chosen provider, keeping it secret.
    Sets it into the process env (and optionally .env) so build_provider finds it.
    """
    if preset.keyless:
        print(f"\n{preset.label} needs no API key.")
        return

    primary = preset.env_names[0]
    existing = resolve_api_key(preset.env_names, allow_prompt=False, project_dir=PROJECT_DIR)

    if existing:
        print(f"\nFound an existing key: {mask(existing.value)} (from {existing.source})")
        if not _ask_yes_no("Use this key?", default=True):
            existing = None  # fall through to entry

    if not existing:
        print(f"\nEnter your {preset.label} API key (input hidden).")
        print(f"  Get one from the provider's console; it will be stored as {primary}.")
        try:
            value = getpass.getpass("API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            raise SystemExit(0)
        if not value:
            print("  No key entered — you'll be prompted again when organizing.")
            return
        if _ask_yes_no("Save this key to .env for next time?", default=True):
            path = save_key_to_dotenv(primary, value, project_dir=PROJECT_DIR)
            print(f"  Saved to {path} (gitignored, permissions 0600). Key not displayed.")
        else:
            # Keep it for this run only.
            import os
            os.environ[primary] = value
            print("  Using the key for this run only (not saved).")


def _choose_model(preset) -> Optional[str]:
    """Let the user pick/confirm the LLM model. Returns None to use the default."""
    default = preset.default_model
    if preset.suggested_models:
        options = [(m, m + (" — default" if m == default else "")) for m in preset.suggested_models]
        options.append(("__custom__", "Enter a different model id"))
        default_index = next(
            (i for i, (m, _) in enumerate(options) if m == default), 0
        )
        choice = _choose("Choose a model:", options, default_index=default_index)
        if choice != "__custom__":
            return choice
        custom = _ask("Model id", default or None)
        return custom or None

    # No suggestions (e.g. custom endpoint): free text.
    label = default or "(required)"
    custom = _ask("Model id", label if default else None)
    return custom or None


def _setup_categories(config: OrganizerConfig) -> None:
    """Pick the category source: built-in defaults, an existing file, or LLM-discovered."""
    choice = _choose(
        "Categories:",
        [
            ("defaults", "Built-in default categories"),
            ("discover", "Auto-discover from these files (uses the LLM)"),
            ("file", "Use an existing category file (JSON/YAML)"),
        ],
        default_index=0,
    )
    if choice == "defaults":
        return
    if choice == "file":
        config.custom_categories_file = _ask("Path to category file")
        return

    # Discover from the chosen input folder, save, and optionally adopt.
    from main import _run_discovery  # local import avoids an import cycle
    out = Path(_ask("Save discovered categories to", "categories.json")).expanduser()
    _run_discovery(config, out)
    if _ask_yes_no("Use these discovered categories now?", default=True):
        config.custom_categories_file = str(out)


# ----------------------------- the wizard ---------------------------------

def run_interactive(base_config: Optional[OrganizerConfig] = None) -> None:
    config = base_config or OrganizerConfig()

    print("=" * 60)
    print("  AI-Powered File Organizer — interactive setup")
    print("=" * 60)

    # 1. Backend
    backend = _choose(
        "Classification backend:",
        [
            ("local", "Local — free, offline, no API key"),
            ("llm", "LLM — bring your own API key (more accurate)"),
        ],
        default_index=0 if config.backend == "local" else 1,
    )
    config.backend = backend

    # 2. Backend-specific configuration
    if backend == "llm":
        provider_options = [(p.key, f"{p.label} ({p.key})") for p in list_presets()]
        default_idx = next(
            (i for i, (k, _) in enumerate(provider_options) if k == config.llm_provider), 0
        )
        provider_key = _choose("LLM provider:", provider_options, default_index=default_idx)
        config.llm_provider = provider_key
        preset = PRESETS[provider_key]

        if provider_key == "openai-compatible":
            config.llm_base_url = _ask(
                "Base URL for the endpoint", config.llm_base_url or "http://localhost:8000/v1"
            )

        _setup_key(preset)
        config.llm_model = _choose_model(preset)

        floor = _ask("Min confidence to file a guess (0-1; 0 = file everything)", "0")
        try:
            config.llm_min_confidence = max(0.0, min(1.0, float(floor)))
        except ValueError:
            config.llm_min_confidence = 0.0
    else:
        config.embedding_model = _choose(
            "Embedding model:",
            [(name, f"{name} — {blurb}") for name, blurb in _EMBED_MODELS],
            default_index=2,
        )

    # 3. Folders
    print()
    in_dir = _ask("Folder to organize", config.input_directories[0] if config.input_directories else "./input")
    while not Path(in_dir).expanduser().is_dir():
        print(f"  '{in_dir}' is not a folder.")
        in_dir = _ask("Folder to organize")
    config.input_directories = [str(Path(in_dir).expanduser())]
    config.output_directory = str(Path(_ask("Destination folder", config.output_directory)).expanduser())

    # 3b. Categories (LLM backend can auto-discover them from these files)
    if backend == "llm":
        _setup_categories(config)

    # 4. Action
    action = _choose(
        "What should I do?",
        [
            ("preview", "Preview — show where files would go (no changes)"),
            ("dry-run", "Dry run — full plan + logs, but don't move anything"),
            ("move", "Organize — move files into the destination"),
            ("copy", "Organize — copy files (originals stay put)"),
        ],
        default_index=0,
    )

    # 5. Build organizer and run
    from main import FileOrganizer  # local import avoids an import cycle

    if action == "preview":
        limit = _ask("How many files to preview?", "10")
        config.dry_run = True
        organizer = FileOrganizer(config)
        _summary(config, "preview")
        organizer.preview(limit=int(limit) if limit.isdigit() else 10)
        return

    config.dry_run = (action == "dry-run")
    config.copy_instead_of_move = (action == "copy")

    _summary(config, action)
    if not config.dry_run and not _ask_yes_no(
        f"This will {'copy' if config.copy_instead_of_move else 'MOVE'} files. Proceed?",
        default=False,
    ):
        print("Aborted — nothing changed.")
        return

    organizer = FileOrganizer(config)
    organizer.organize(show_progress=True, assume_yes=True)


def _summary(config: OrganizerConfig, action: str) -> None:
    print("\n" + "-" * 60)
    print("Summary")
    print("-" * 60)
    print(f"  Backend : {config.backend}")
    if config.backend == "llm":
        print(f"  Provider: {config.llm_provider}")
        print(f"  Model   : {config.llm_model or '(provider default)'}")
        if config.llm_base_url:
            print(f"  Endpoint: {config.llm_base_url}")
        if config.llm_min_confidence > 0:
            print(f"  Review  : files below {config.llm_min_confidence:.0%} confidence")
    else:
        print(f"  Model   : {config.embedding_model}")
    print(f"  Categories: {config.custom_categories_file or 'built-in defaults'}")
    print(f"  Input   : {config.input_directories[0]}")
    print(f"  Output  : {config.output_directory}")
    print(f"  Action  : {action}")
    print("-" * 60)
