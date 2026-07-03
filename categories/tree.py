# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Hierarchical category tree structure.
Each node has a name, description (for embedding), confidence threshold, and children.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CategoryNode:
    """A node in the category tree."""
    name: str
    description: str  # Used for embedding - describe what files belong here
    threshold: float = 0.5  # Minimum similarity to classify into this category
    children: list['CategoryNode'] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # Computed at runtime
    
    def add_child(self, child: 'CategoryNode') -> 'CategoryNode':
        """Add a child category and return it."""
        self.children.append(child)
        return child
    
    def get_child_by_name(self, name: str) -> Optional['CategoryNode']:
        """Find a direct child by name."""
        for child in self.children:
            if child.name.lower() == name.lower():
                return child
        return None
    
    def has_other_folder(self) -> bool:
        """Check if this node has an 'Other' child."""
        return self.get_child_by_name("Other") is not None
    
    def ensure_other_folder(self) -> 'CategoryNode':
        """Ensure this category has an 'Other' subfolder, create if missing."""
        other = self.get_child_by_name("Other")
        if other is None:
            other = CategoryNode(
                name="Other",
                description=f"Files related to {self.name} that don't fit specific subcategories",
                threshold=0.0  # Always accepts if parent matched
            )
            self.children.append(other)
        return other
    
    def get_path(self, parent_path: str = "") -> str:
        """Get the full path for this category."""
        if parent_path:
            return f"{parent_path}/{self.name}"
        return self.name
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CategoryNode':
        """Create from dictionary."""
        node = cls(
            name=data["name"],
            description=data["description"],
            threshold=data.get("threshold", 0.5)
        )
        for child_data in data.get("children", []):
            node.children.append(cls.from_dict(child_data))
        return node
    
    def __repr__(self) -> str:
        return f"CategoryNode(name='{self.name}', children={len(self.children)})"


class CategoryTree:
    """Manages the full category hierarchy."""
    
    def __init__(self):
        self.root_categories: list[CategoryNode] = []
        self._other_category: Optional[CategoryNode] = None
    
    def add_root_category(self, category: CategoryNode) -> None:
        """Add a top-level category."""
        self.root_categories.append(category)
    
    @property
    def other_category(self) -> CategoryNode:
        """Get or create the root 'Other' category."""
        if self._other_category is None:
            self._other_category = CategoryNode(
                name="Other",
                description="Files that don't match any known category",
                threshold=0.0
            )
        return self._other_category
    
    def ensure_all_other_folders(self) -> None:
        """Recursively ensure every category has an 'Other' subfolder."""
        def add_other_recursive(node: CategoryNode):
            if node.children and node.name != "Other":
                node.ensure_other_folder()
                for child in node.children:
                    if child.name != "Other":
                        add_other_recursive(child)
        
        for category in self.root_categories:
            add_other_recursive(category)
    
    def get_all_nodes(self) -> list[CategoryNode]:
        """Get all nodes in the tree (flat list)."""
        nodes = []
        
        def collect(node: CategoryNode):
            nodes.append(node)
            for child in node.children:
                collect(child)
        
        for category in self.root_categories:
            collect(category)
        nodes.append(self.other_category)
        return nodes
    
    def get_all_leaf_paths(self) -> list[tuple[str, CategoryNode]]:
        """Get all possible destination paths with their nodes."""
        paths = []
        
        def collect(node: CategoryNode, current_path: str):
            path = f"{current_path}/{node.name}" if current_path else node.name
            if not node.children:
                paths.append((path, node))
            else:
                for child in node.children:
                    collect(child, path)
        
        for category in self.root_categories:
            collect(category, "")
        paths.append((self.other_category.name, self.other_category))
        return paths
    
    def to_dict(self) -> dict:
        """Serialize the tree."""
        return {
            "categories": [cat.to_dict() for cat in self.root_categories]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CategoryTree':
        """Deserialize the tree."""
        tree = cls()
        for cat_data in data.get("categories", []):
            tree.add_root_category(CategoryNode.from_dict(cat_data))
        return tree

    @classmethod
    def from_file(cls, path) -> 'CategoryTree':
        """
        Load a category tree from a JSON or YAML file and ensure every branch
        has an 'Other' fallback. Accepts either {"categories": [...]} or a bare
        list of category dicts.
        """
        from pathlib import Path
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yml", ".yaml"):
            import yaml
            data = yaml.safe_load(text)
        else:
            import json
            data = json.loads(text)

        if isinstance(data, list):
            data = {"categories": data}
        if not isinstance(data, dict) or not data.get("categories"):
            raise ValueError(
                f"Category file {path} must contain a non-empty 'categories' list."
            )

        tree = cls.from_dict(data)
        tree.ensure_all_other_folders()
        return tree
    
    def print_tree(self, indent: int = 0) -> None:
        """Pretty print the category tree."""
        def print_node(node: CategoryNode, level: int):
            prefix = "  " * level + ("├── " if level > 0 else "")
            threshold_str = f" (threshold: {node.threshold:.2f})" if node.threshold > 0 else ""
            print(f"{prefix}{node.name}{threshold_str}")
            for child in node.children:
                print_node(child, level + 1)
        
        print("Category Tree:")
        print("=" * 40)
        for category in self.root_categories:
            print_node(category, 0)
        print_node(self.other_category, 0)
