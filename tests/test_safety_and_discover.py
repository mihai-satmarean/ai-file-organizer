# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""Output-inside-input guard and LLM category discovery normalization."""

import pytest

from main import _guard_output_not_in_input
from llm.discover import _clean_node, _sample, discover_categories
from categories.tree import CategoryTree

from conftest import FakeProvider


# ---- safety guard --------------------------------------------------------

def test_guard_allows_separate_dirs(tmp_path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir(); out.mkdir()
    _guard_output_not_in_input([inp], out)  # no exception


def test_guard_blocks_output_inside_input(tmp_path):
    inp = tmp_path / "in"
    (inp / "organized").mkdir(parents=True)
    with pytest.raises(ValueError):
        _guard_output_not_in_input([inp], inp / "organized")


def test_guard_blocks_equal(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    with pytest.raises(ValueError):
        _guard_output_not_in_input([d], d)


# ---- discovery normalization ---------------------------------------------

def test_clean_node_sanitizes_and_drops_other():
    assert _clean_node({"name": "Other", "description": "x"}) is None
    node = _clean_node({"name": "Sch/ool:Work", "description": "d"})
    assert "/" not in node["name"] and ":" not in node["name"]
    assert node["threshold"] == 0.5


def test_clean_node_depth_cap():
    deep = {"name": "L0", "description": "d", "children": [
        {"name": "L1", "description": "d", "children": [
            {"name": "L2", "description": "d", "children": [
                {"name": "L3", "description": "d"}]}]}]}
    out = _clean_node(deep)
    # 3 levels kept (L0>L1>L2), L3 dropped
    assert out["children"][0]["children"][0]["name"] == "L2"
    assert "children" not in out["children"][0]["children"][0]


def test_sample_caps_count():
    assert len(_sample(list(range(100)), 10)) == 10
    assert _sample([1, 2, 3], 10) == [1, 2, 3]


def test_discover_categories_builds_valid_tree():
    payload = {"categories": [
        {"name": "Photos", "description": "images", "children": [
            {"name": "Screenshots", "description": "screens"}]},
        {"name": "Other", "description": "ignored"},  # must be dropped
    ]}
    from pathlib import Path
    from extractors.base import ExtractedContent
    contents = [ExtractedContent(Path("a.png"), "a.png", ".png", "", {}, True)]
    result = discover_categories(FakeProvider(payload), contents)
    names = [c["name"] for c in result["categories"]]
    assert names == ["Photos"]  # 'Other' dropped
    # Round-trips into a real tree.
    tree = CategoryTree.from_dict(result)
    assert tree.root_categories[0].name == "Photos"
