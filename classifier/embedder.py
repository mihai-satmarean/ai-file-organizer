# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright (c) 2026 George Zhang — https://github.com/TheYellowDuck

"""
Embedding model wrapper using sentence-transformers.
Generates embeddings for text content and category descriptions.
"""

from typing import Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class Embedder:
    """
    Wrapper for the sentence-transformers embedding model.
    Uses all-MiniLM-L6-v2 by default - a good balance of speed and quality.
    """
    
    # Default model - fast and effective
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    # Alternative models (uncomment to try):
    # "all-mpnet-base-v2"  # Higher quality, slower
    # "paraphrase-MiniLM-L6-v2"  # Good for paraphrase detection
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to all-MiniLM-L6-v2.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
    
    @property
    def model(self) -> 'SentenceTransformer':
        """Lazy load the model on first use."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            # Generate a test embedding to get dimension
            test_embedding = self.embed("test")
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Numpy array of the embedding vector.
        """
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed.
            show_progress: Whether to show a progress bar.
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        # Handle empty texts
        processed_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if text.strip():
                processed_texts.append(text)
            else:
                empty_indices.append(i)
                processed_texts.append("empty")  # Placeholder
        
        embeddings = self.model.encode(
            processed_texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        # Zero out embeddings for empty texts
        for i in empty_indices:
            embeddings[i] = np.zeros(embeddings.shape[1])
        
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score between -1 and 1.
        """
        # Handle zero vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def similarity_to_many(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity of one embedding to many candidates.
        
        Args:
            query_embedding: Single embedding vector.
            candidate_embeddings: Array of candidate embeddings.
            
        Returns:
            Array of similarity scores.
        """
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(candidate_embeddings))
        
        query_normalized = query_embedding / query_norm
        
        # Normalize candidates
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        candidate_norms = np.where(candidate_norms == 0, 1, candidate_norms)
        candidates_normalized = candidate_embeddings / candidate_norms
        
        # Compute similarities
        similarities = np.dot(candidates_normalized, query_normalized)
        
        return similarities
