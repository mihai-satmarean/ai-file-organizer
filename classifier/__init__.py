# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""Classification package for the file organizer."""

from classifier.embedder import Embedder
from classifier.hierarchical import HierarchicalClassifier, ClassificationResult

__all__ = [
    'Embedder',
    'HierarchicalClassifier',
    'ClassificationResult',
]
