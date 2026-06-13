"""
Configuration management for the file organizer.

Security note: this config is the kind of thing users commit to disk, so it
NEVER holds an API key. Keys are resolved at runtime from the environment, a
.env file, or an interactive prompt (see llm/secrets.py). Any `api_key`-looking
field found in a loaded config is dropped with a warning.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional
import yaml
import json


# Field names we refuse to read from / write to a config file, to avoid ever
# persisting a secret alongside non-sensitive settings.
_SECRET_FIELD_NAMES = {"api_key", "apikey", "api_key_value", "token", "secret"}


@dataclass
class OrganizerConfig:
    """Configuration for the file organizer."""

    # Input/Output paths
    input_directories: list[str] = field(default_factory=lambda: ["./input"])
    output_directory: str = "./organized"

    # Scanning options
    recursive_scan: bool = True
    skip_hidden_files: bool = True
    max_file_size_mb: float = 100.0
    exclude_patterns: list[str] = field(default_factory=list)  # globs from --exclude / .organizerignore

    # Privacy: warn before sending file contents to a third-party LLM API
    show_data_notice: bool = True

    # Classification options
    enable_ocr: bool = False  # OCR is slow, disabled by default

    # Backend: "local" (free, offline embeddings) or "llm" (bring-your-own-key).
    backend: str = "local"

    # Embedding model (used when backend == "local"):
    # Fast:     "all-MiniLM-L6-v2"      (~90MB, fastest)
    # Balanced: "all-MiniLM-L12-v2"     (~120MB, better accuracy)
    # Accurate: "all-mpnet-base-v2"     (~420MB, best accuracy, slower)
    embedding_model: str = "all-mpnet-base-v2"

    # LLM backend options (used when backend == "llm"). NO api key here —
    # keys come from the environment / .env / prompt only.
    llm_provider: str = "anthropic"        # any key in llm.providers.PRESETS
    llm_model: Optional[str] = None         # None -> provider's default model
    llm_base_url: Optional[str] = None      # override endpoint (custom vendors)
    llm_batch_size: int = 20                # files classified per API request
    llm_max_chars_per_file: int = 1200      # per-file content sent to the model
    llm_auto_install_sdk: bool = True       # install the provider's SDK on demand
    llm_min_confidence: float = 0.0         # below this, files go to the Review folder (0 = off)
    show_token_usage: bool = True           # print LLM token/cache totals after a run
    llm_vision: bool = True                  # send image files to the model as images
    llm_vision_max_dim: int = 1024           # downscale images to this max edge (px)
    llm_max_files: int = 2000                # refuse LLM runs larger than this (0 = no cap)
    llm_concurrency: int = 4                  # parallel classification requests
    llm_max_retries: int = 4                  # SDK retries (exp. backoff) on 429/5xx

    # Organization options
    dry_run: bool = False
    copy_instead_of_move: bool = False
    handle_duplicates: str = "rename"  # "rename", "skip", "overwrite"
    create_logs: bool = True
    create_undo_script: bool = True

    # Category options
    use_default_categories: bool = True
    custom_categories_file: Optional[str] = None
    auto_discover_categories: bool = False  # Future feature

    @classmethod
    def _from_dict(cls, data: dict) -> 'OrganizerConfig':
        """Build a config from a dict, ignoring unknown keys and dropping secrets."""
        known = {f.name for f in fields(cls)}
        clean: dict = {}
        for key, value in data.items():
            if key.lower() in _SECRET_FIELD_NAMES:
                print(
                    f"⚠️  Ignoring '{key}' in config file — API keys must not be "
                    "stored in config. Use an environment variable or .env file."
                )
                continue
            if key in known:
                clean[key] = value
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: Path) -> 'OrganizerConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: Path) -> 'OrganizerConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    def _safe_dict(self) -> dict:
        """Serializable view of the config with any secret-looking fields stripped."""
        return {
            k: v for k, v in self.__dict__.items()
            if k.lower() not in _SECRET_FIELD_NAMES
        }

    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file (never includes secrets)."""
        with open(path, 'w') as f:
            yaml.dump(self._safe_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Path) -> None:
        """Save config to JSON file (never includes secrets)."""
        with open(path, 'w') as f:
            json.dump(self._safe_dict(), f, indent=2)


def create_default_config_file(path: Path) -> None:
    """Create a default configuration file."""
    config = OrganizerConfig()

    if path.suffix in ('.yml', '.yaml'):
        config.to_yaml(path)
    else:
        config.to_json(path)

    print(f"Created default config file: {path}")
