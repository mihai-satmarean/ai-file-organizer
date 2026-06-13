"""
Hierarchical classifier for categorizing files.
Uses embeddings to match file content to category descriptions.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from classifier.embedder import Embedder
from categories.tree import CategoryNode, CategoryTree
from extractors.base import ExtractedContent


@dataclass
class ClassificationResult:
    """Result of classifying a single file."""
    file_path: str
    category_path: str  # e.g., "School/University/CS246"
    confidence_scores: dict[str, float]  # Score at each level
    final_confidence: float  # Confidence of the final classification
    is_other: bool  # Whether it ended up in an "Other" folder
    classification_chain: list[str]  # The categories matched at each level


class HierarchicalClassifier:
    """
    Classifies files into a hierarchical category structure.
    Uses embedding similarity to match content to categories.
    """
    
    def __init__(
        self, 
        category_tree: CategoryTree,
        embedder: Optional[Embedder] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            category_tree: The category hierarchy to classify into.
            embedder: Embedding model to use. Creates default if not provided.
        """
        self.category_tree = category_tree
        self.embedder = embedder or Embedder()
        self._categories_embedded = False
    
    def _embed_categories(self) -> None:
        """Pre-compute embeddings for all category descriptions."""
        if self._categories_embedded:
            return
        
        print("Embedding category descriptions...")
        
        def embed_recursive(node: CategoryNode):
            node.embedding = self.embedder.embed(node.description)
            for child in node.children:
                embed_recursive(child)
        
        for category in self.category_tree.root_categories:
            embed_recursive(category)
        
        # Embed the root "Other" category
        self.category_tree.other_category.embedding = self.embedder.embed(
            self.category_tree.other_category.description
        )
        
        self._categories_embedded = True
        print("Category embeddings ready!")
    
    def classify(self, content: ExtractedContent) -> ClassificationResult:
        """
        Classify a file into the category hierarchy.
        
        Args:
            content: Extracted content from the file.
            
        Returns:
            ClassificationResult with the determined category path.
        """
        # Ensure categories are embedded
        self._embed_categories()
        
        # Get combined content (filename + text content)
        text_to_classify = content.combined_content
        
        # Embed the file content
        file_embedding = self.embedder.embed(text_to_classify)
        
        # Classify hierarchically
        return self._classify_hierarchically(
            file_embedding=file_embedding,
            file_path=str(content.file_path)
        )
    
    def _classify_hierarchically(
        self,
        file_embedding: np.ndarray,
        file_path: str
    ) -> ClassificationResult:
        """
        Recursively classify through the category hierarchy.
        """
        confidence_scores = {}
        classification_chain = []
        current_path = []
        
        # First, try to match top-level categories
        root_categories = self.category_tree.root_categories
        best_root, best_root_score = self._find_best_match(
            file_embedding, 
            root_categories
        )
        
        confidence_scores["root"] = best_root_score
        
        # Check if it meets the threshold
        if best_root is None or best_root_score < best_root.threshold:
            # Doesn't match any top-level category -> goes to root Other
            return ClassificationResult(
                file_path=file_path,
                category_path="Other",
                confidence_scores=confidence_scores,
                final_confidence=best_root_score if best_root else 0.0,
                is_other=True,
                classification_chain=["Other"]
            )
        
        # Matched a root category
        current_node = best_root
        current_path.append(current_node.name)
        classification_chain.append(current_node.name)
        
        # Recursively classify into subcategories
        level = 1
        while current_node.children:
            # Find best matching child
            # Exclude "Other" from initial matching - it's the fallback
            non_other_children = [c for c in current_node.children if c.name != "Other"]
            
            if not non_other_children:
                # Only "Other" exists as child
                break
            
            best_child, best_child_score = self._find_best_match(
                file_embedding,
                non_other_children
            )
            
            confidence_scores[f"level_{level}"] = best_child_score
            
            if best_child is None or best_child_score < best_child.threshold:
                # Doesn't match any subcategory -> goes to "Other" at this level
                other_child = current_node.get_child_by_name("Other")
                if other_child:
                    current_path.append("Other")
                    classification_chain.append("Other")
                break
            
            # Matched a subcategory
            current_path.append(best_child.name)
            classification_chain.append(best_child.name)
            current_node = best_child
            level += 1
        
        # Build final path
        category_path = "/".join(current_path)
        is_other = current_path[-1] == "Other" if current_path else True
        
        # Final confidence is the minimum confidence along the chain
        final_confidence = min(confidence_scores.values()) if confidence_scores else 0.0
        
        return ClassificationResult(
            file_path=file_path,
            category_path=category_path,
            confidence_scores=confidence_scores,
            final_confidence=final_confidence,
            is_other=is_other,
            classification_chain=classification_chain
        )
    
    def _find_best_match(
        self,
        file_embedding: np.ndarray,
        categories: list[CategoryNode]
    ) -> tuple[Optional[CategoryNode], float]:
        """
        Find the best matching category from a list.
        
        Returns:
            Tuple of (best_category, similarity_score)
        """
        if not categories:
            return None, 0.0
        
        best_category = None
        best_score = -1.0
        
        for category in categories:
            if category.embedding is None:
                continue
            
            score = self.embedder.similarity(file_embedding, category.embedding)
            
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category, best_score
    
    def classify_batch(
        self, 
        contents: list[ExtractedContent],
        show_progress: bool = True
    ) -> list[ClassificationResult]:
        """
        Classify multiple files efficiently.
        
        Args:
            contents: List of extracted file contents.
            show_progress: Whether to show progress.
            
        Returns:
            List of classification results.
        """
        self._embed_categories()
        
        if show_progress:
            from tqdm import tqdm
            contents = tqdm(contents, desc="Classifying files")
        
        results = []
        for content in contents:
            result = self.classify(content)
            results.append(result)
        
        return results
    
    def explain_classification(self, result: ClassificationResult) -> str:
        """
        Generate a human-readable explanation of a classification.
        """
        lines = [
            f"File: {result.file_path}",
            f"Classified to: {result.category_path}",
            f"Overall confidence: {result.final_confidence:.2%}",
            "",
            "Classification chain:"
        ]
        
        for i, category in enumerate(result.classification_chain):
            level_key = "root" if i == 0 else f"level_{i}"
            score = result.confidence_scores.get(level_key, 0)
            indent = "  " * i
            lines.append(f"{indent}→ {category} (confidence: {score:.2%})")
        
        if result.is_other:
            lines.append("")
            lines.append("Note: File ended up in 'Other' folder (low confidence match)")
        
        return "\n".join(lines)
