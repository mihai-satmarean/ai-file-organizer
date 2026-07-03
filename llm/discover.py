# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
LLM-driven category discovery.

Given a sample of the user's files, ask the model to propose a clean,
general-purpose hierarchical category tree (2-3 levels) for organizing them.
The result is returned as a plain dict in the same shape CategoryTree.from_dict
expects, so it can be saved to a file and reused with --categories-file.
"""

from __future__ import annotations

from typing import Optional

from extractors.base import ExtractedContent
from llm.providers import BaseProvider


_FOLDER_SAFE = '<>:"/\\|?*'


def _sample(items: list, n: int) -> list:
    """Evenly spread a sample of up to n items across the list for variety."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _sanitize_name(name: str) -> str:
    name = "".join(c for c in str(name) if c not in _FOLDER_SAFE).strip()
    return name[:64] or "Untitled"


def _clean_node(node: dict, depth: int = 0) -> Optional[dict]:
    """Validate/normalize one proposed category node; drop 'Other' (auto-added)."""
    if not isinstance(node, dict):
        return None
    name = _sanitize_name(node.get("name", ""))
    if not name or name.lower() == "other":
        return None
    out = {
        "name": name,
        "description": str(node.get("description", name))[:300],
        "threshold": 0.5,
    }
    children = []
    if depth < 2:  # cap at 3 levels total
        for child in node.get("children", []) or []:
            cleaned = _clean_node(child, depth + 1)
            if cleaned:
                children.append(cleaned)
    if children:
        out["children"] = children
    return out


def discover_categories(
    provider: BaseProvider,
    contents: list[ExtractedContent],
    *,
    max_files: int = 40,
    max_chars_per_file: int = 400,
) -> dict:
    """
    Propose a category tree from a sample of files.

    Returns a dict like {"categories": [ {name, description, threshold,
    children?}, ... ]} suitable for CategoryTree.from_dict / saving to a file.
    """
    sample = _sample(contents, max_files)

    listing = []
    for c in sample:
        snippet = " ".join(c.combined_content[:max_chars_per_file].split())
        listing.append(f"- {c.file_path.name}: {snippet}")
    files_block = "\n".join(listing)

    system = (
        "You design folder taxonomies for organizing files. Given a sample of "
        "files (names + content snippets), propose a clean, GENERAL-PURPOSE "
        "hierarchical category tree to sort them into.\n"
        "Guidelines:\n"
        "- 3 to 7 broad top-level categories; nest 1-2 levels where useful (max 3 deep).\n"
        "- Names must be short and folder-safe (letters, numbers, spaces).\n"
        "- Prefer reusable categories over hyper-specific ones.\n"
        "- Do NOT add 'Other' folders — they are added automatically.\n"
        "- Each node needs a one-sentence 'description' of what belongs there.\n\n"
        "Respond with ONLY JSON of the form:\n"
        '{"categories": [{"name": "...", "description": "...", '
        '"children": [{"name": "...", "description": "..."}]}]}'
    )
    user = f"Here is a sample of {len(sample)} files:\n\n{files_block}"

    data = provider.complete_json(system, user, schema=None, max_tokens=4096)

    raw = data.get("categories", []) if isinstance(data, dict) else []
    categories = [n for n in (_clean_node(c) for c in raw) if n]
    if not categories:
        raise ValueError("The model did not return a usable category tree.")
    return {"categories": categories}
