"""Category management for the file organizer."""

from categories.tree import CategoryNode, CategoryTree
from categories.default_categories import (
    create_default_categories,
    create_minimal_categories,
    create_waterloo_example_categories,
)

__all__ = [
    'CategoryNode',
    'CategoryTree',
    'create_default_categories',
    'create_minimal_categories',
    'create_waterloo_example_categories',
]
