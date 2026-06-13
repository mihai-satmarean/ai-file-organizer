"""Category tree (de)serialization and the custom-file loader."""

import json

import pytest

from categories.tree import CategoryTree


SAMPLE = {
    "categories": [
        {
            "name": "School",
            "description": "school stuff",
            "children": [
                {"name": "Math", "description": "math"},
                {"name": "Science", "description": "science"},
            ],
        },
        {"name": "Work", "description": "work stuff"},
    ]
}


def test_from_dict_roundtrip():
    tree = CategoryTree.from_dict(SAMPLE)
    names = [c.name for c in tree.root_categories]
    assert names == ["School", "Work"]
    out = tree.to_dict()
    assert out["categories"][0]["children"][0]["name"] == "Math"


def test_from_file_json_adds_other_folders(tmp_path):
    p = tmp_path / "cats.json"
    p.write_text(json.dumps(SAMPLE))
    tree = CategoryTree.from_file(p)
    leaves = {path for path, _ in tree.get_all_leaf_paths()}
    assert "School/Math" in leaves
    # ensure_all_other_folders() added an Other under the branch with children.
    assert "School/Other" in leaves


def test_from_file_yaml(tmp_path):
    p = tmp_path / "cats.yaml"
    p.write_text(
        "categories:\n"
        "  - name: Personal\n"
        "    description: personal\n"
    )
    tree = CategoryTree.from_file(p)
    assert [c.name for c in tree.root_categories] == ["Personal"]


def test_from_file_bare_list(tmp_path):
    p = tmp_path / "cats.json"
    p.write_text(json.dumps([{"name": "A", "description": "a"}]))
    tree = CategoryTree.from_file(p)
    assert tree.root_categories[0].name == "A"


def test_from_file_empty_raises(tmp_path):
    p = tmp_path / "cats.json"
    p.write_text(json.dumps({"categories": []}))
    with pytest.raises(ValueError):
        CategoryTree.from_file(p)
