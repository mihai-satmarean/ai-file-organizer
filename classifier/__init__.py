"""Classification package for the file organizer."""

from classifier.embedder import Embedder
from classifier.hierarchical import HierarchicalClassifier, ClassificationResult

__all__ = [
    'Embedder',
    'HierarchicalClassifier',
    'ClassificationResult',
]
