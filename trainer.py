# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Training and calibration module for improving classification accuracy.

This module allows you to:
1. Provide example files for each category
2. Automatically tune thresholds based on examples
3. See detailed diagnostics on why files are classified certain ways
4. Build a "memory" of corrections to improve future classification
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from categories.tree import CategoryNode, CategoryTree
from extractors.registry import ExtractorRegistry
from classifier.embedder import Embedder


@dataclass
class TrainingExample:
    """A single training example."""
    file_path: str
    correct_category: str  # e.g., "School/University/CS 246"
    embedding: Optional[np.ndarray] = None


@dataclass 
class CategoryTrainingData:
    """Training data for a single category."""
    category_path: str
    examples: list[TrainingExample] = field(default_factory=list)
    mean_embedding: Optional[np.ndarray] = None
    
    def compute_mean_embedding(self):
        """Compute average embedding from examples."""
        if not self.examples:
            return
        embeddings = [ex.embedding for ex in self.examples if ex.embedding is not None]
        if embeddings:
            self.mean_embedding = np.mean(embeddings, axis=0)


class Trainer:
    """
    Trains and calibrates the classifier using example files.
    """
    
    def __init__(
        self,
        category_tree: CategoryTree,
        embedder: Optional[Embedder] = None,
        extractor: Optional[ExtractorRegistry] = None
    ):
        self.category_tree = category_tree
        self.embedder = embedder or Embedder()
        self.extractor = extractor or ExtractorRegistry(enable_ocr=False)
        
        self.training_data: dict[str, CategoryTrainingData] = {}
        self.corrections: list[dict] = []  # History of user corrections
        
    def add_example(self, file_path: Path, category_path: str) -> None:
        """
        Add a training example for a category.
        
        Args:
            file_path: Path to an example file
            category_path: The correct category, e.g., "School/University/CS 246"
        """
        file_path = Path(file_path)
        
        # Extract content and embed
        content = self.extractor.extract(file_path)
        embedding = self.embedder.embed(content.combined_content)
        
        example = TrainingExample(
            file_path=str(file_path),
            correct_category=category_path,
            embedding=embedding
        )
        
        # Add to training data
        if category_path not in self.training_data:
            self.training_data[category_path] = CategoryTrainingData(category_path=category_path)
        
        self.training_data[category_path].examples.append(example)
        self.training_data[category_path].compute_mean_embedding()
        
        print(f"Added example: {file_path.name} → {category_path}")
    
    def add_examples_from_folder(self, folder_path: Path, category_path: str) -> int:
        """
        Add all files in a folder as examples for a category.
        
        Args:
            folder_path: Folder containing example files
            category_path: The category these files belong to
            
        Returns:
            Number of examples added
        """
        folder_path = Path(folder_path)
        count = 0
        
        for file_path in folder_path.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    self.add_example(file_path, category_path)
                    count += 1
                except Exception as e:
                    print(f"  Skipped {file_path.name}: {e}")
        
        return count
    
    def record_correction(
        self, 
        file_path: Path, 
        predicted_category: str, 
        correct_category: str
    ) -> None:
        """
        Record a correction when the classifier was wrong.
        This builds a history that can be used to improve accuracy.
        """
        content = self.extractor.extract(file_path)
        embedding = self.embedder.embed(content.combined_content)
        
        correction = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "predicted": predicted_category,
            "correct": correct_category,
            "embedding": embedding.tolist()
        }
        
        self.corrections.append(correction)
        
        # Also add as a training example
        self.add_example(file_path, correct_category)
        
        print(f"Recorded correction: {file_path.name}")
        print(f"  Was: {predicted_category}")
        print(f"  Should be: {correct_category}")
    
    def diagnose_file(self, file_path: Path) -> dict:
        """
        Get detailed diagnostics on how a file would be classified.
        Useful for understanding why misclassification happens.
        """
        file_path = Path(file_path)
        content = self.extractor.extract(file_path)
        file_embedding = self.embedder.embed(content.combined_content)
        
        # Compute similarity to all categories
        similarities = {}
        
        def compute_similarities(node: CategoryNode, path: str = ""):
            current_path = f"{path}/{node.name}" if path else node.name
            
            if node.embedding is not None:
                sim = self.embedder.similarity(file_embedding, node.embedding)
                similarities[current_path] = {
                    "similarity": sim,
                    "threshold": node.threshold,
                    "passes_threshold": sim >= node.threshold,
                    "description": node.description[:100] + "..."
                }
            
            for child in node.children:
                compute_similarities(child, current_path)
        
        # Make sure categories are embedded
        self._ensure_categories_embedded()
        
        for category in self.category_tree.root_categories:
            compute_similarities(category)
        
        # Sort by similarity
        sorted_sims = sorted(
            similarities.items(), 
            key=lambda x: x[1]["similarity"], 
            reverse=True
        )
        
        return {
            "file": str(file_path),
            "file_name": file_path.name,
            "content_preview": content.combined_content[:200],
            "top_matches": sorted_sims[:10],
            "all_similarities": similarities
        }
    
    def print_diagnosis(self, file_path: Path) -> None:
        """Print a human-readable diagnosis for a file."""
        diag = self.diagnose_file(file_path)
        
        print(f"\n{'='*60}")
        print(f"Diagnosis for: {diag['file_name']}")
        print(f"{'='*60}")
        print(f"\nContent preview:")
        print(f"  {diag['content_preview'][:100]}...")
        print(f"\nTop 10 category matches:")
        print(f"{'-'*60}")
        
        for path, data in diag['top_matches']:
            status = "✓" if data['passes_threshold'] else "✗"
            print(f"  {status} {path}")
            print(f"      Similarity: {data['similarity']:.3f} (threshold: {data['threshold']})")
        
        print()
    
    def suggest_threshold_adjustments(self) -> dict[str, float]:
        """
        Based on training examples, suggest optimal thresholds for each category.
        """
        suggestions = {}
        
        self._ensure_categories_embedded()
        
        for category_path, data in self.training_data.items():
            if not data.examples:
                continue
            
            # Find the category node
            node = self._find_category_node(category_path)
            if node is None or node.embedding is None:
                continue
            
            # Compute similarities for all examples
            similarities = []
            for example in data.examples:
                if example.embedding is not None:
                    sim = self.embedder.similarity(example.embedding, node.embedding)
                    similarities.append(sim)
            
            if similarities:
                # Suggest threshold at 90th percentile of example similarities
                # This ensures most examples would pass
                min_sim = min(similarities)
                suggested = min_sim * 0.95  # 5% buffer below minimum
                
                suggestions[category_path] = {
                    "current_threshold": node.threshold,
                    "suggested_threshold": round(suggested, 3),
                    "example_similarities": {
                        "min": round(min(similarities), 3),
                        "max": round(max(similarities), 3),
                        "mean": round(np.mean(similarities), 3)
                    },
                    "num_examples": len(similarities)
                }
        
        return suggestions
    
    def apply_threshold_suggestions(self, suggestions: dict) -> None:
        """Apply suggested thresholds to the category tree."""
        for category_path, data in suggestions.items():
            node = self._find_category_node(category_path)
            if node:
                old_threshold = node.threshold
                node.threshold = data["suggested_threshold"]
                print(f"{category_path}: {old_threshold} → {data['suggested_threshold']}")
    
    def _ensure_categories_embedded(self) -> None:
        """Ensure all categories have embeddings."""
        def embed_recursive(node: CategoryNode):
            if node.embedding is None:
                node.embedding = self.embedder.embed(node.description)
            for child in node.children:
                embed_recursive(child)
        
        for category in self.category_tree.root_categories:
            embed_recursive(category)
    
    def _find_category_node(self, category_path: str) -> Optional[CategoryNode]:
        """Find a category node by its path."""
        parts = category_path.split("/")
        
        # Find root category
        current = None
        for cat in self.category_tree.root_categories:
            if cat.name == parts[0]:
                current = cat
                break
        
        if current is None:
            return None
        
        # Navigate to the target
        for part in parts[1:]:
            found = False
            for child in current.children:
                if child.name == part:
                    current = child
                    found = True
                    break
            if not found:
                return None
        
        return current
    
    def save(self, path: Path) -> None:
        """Save training data to a file."""
        data = {
            "corrections": self.corrections,
            "training_data": {
                cat_path: {
                    "category_path": td.category_path,
                    "examples": [
                        {
                            "file_path": ex.file_path,
                            "correct_category": ex.correct_category,
                            "embedding": ex.embedding.tolist() if ex.embedding is not None else None
                        }
                        for ex in td.examples
                    ]
                }
                for cat_path, td in self.training_data.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved training data to {path}")
    
    def load(self, path: Path) -> None:
        """Load training data from a file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.corrections = data.get("corrections", [])
        
        for cat_path, td_data in data.get("training_data", {}).items():
            examples = []
            for ex_data in td_data.get("examples", []):
                embedding = None
                if ex_data.get("embedding"):
                    embedding = np.array(ex_data["embedding"])
                examples.append(TrainingExample(
                    file_path=ex_data["file_path"],
                    correct_category=ex_data["correct_category"],
                    embedding=embedding
                ))
            
            self.training_data[cat_path] = CategoryTrainingData(
                category_path=cat_path,
                examples=examples
            )
            self.training_data[cat_path].compute_mean_embedding()
        
        print(f"Loaded training data from {path}")
        print(f"  {len(self.corrections)} corrections")
        print(f"  {sum(len(td.examples) for td in self.training_data.values())} examples")


def interactive_training_session(trainer: Trainer, files_to_review: list[Path]) -> None:
    """
    Run an interactive session to review and correct classifications.
    """
    from classifier.hierarchical import HierarchicalClassifier
    
    classifier = HierarchicalClassifier(
        category_tree=trainer.category_tree,
        embedder=trainer.embedder
    )
    
    print("\n" + "="*60)
    print("Interactive Training Session")
    print("="*60)
    print("For each file, review the classification and correct if needed.")
    print("Commands: [Enter] = correct, [c] = change category, [s] = skip, [q] = quit\n")
    
    for file_path in files_to_review:
        content = trainer.extractor.extract(file_path)
        result = classifier.classify(content)
        
        print(f"\nFile: {file_path.name}")
        print(f"Predicted: {result.category_path} (confidence: {result.final_confidence:.1%})")
        
        response = input("Is this correct? [Enter/c/s/q]: ").strip().lower()
        
        if response == 'q':
            break
        elif response == 's':
            continue
        elif response == 'c':
            print("\nAvailable categories:")
            trainer.category_tree.print_tree()
            correct_category = input("Enter correct category path: ").strip()
            if correct_category:
                trainer.record_correction(file_path, result.category_path, correct_category)
        else:
            # Correct - add as training example
            trainer.add_example(file_path, result.category_path)
    
    print("\nSession complete!")
    print(f"Training data now has {sum(len(td.examples) for td in trainer.training_data.values())} examples")
