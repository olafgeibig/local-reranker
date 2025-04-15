# -*- coding: utf-8 -*-
"""Handles the model loading and reranking logic."""

import logging
from typing import List, Union, Dict, Any, Optional, Tuple

import torch
from sentence_transformers import CrossEncoder

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Default Model --- 
# Can be overridden during initialization or potentially via API request in future
DEFAULT_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

class Reranker:
    """Loads a CrossEncoder model and performs reranking."""
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        """
        Initializes the Reranker.

        Args:
            model_name: The name of the CrossEncoder model to load 
                        (e.g., 'jinaai/jina-reranker-v2-base-multilingual').
            device: The device to run the model on ('cpu', 'cuda', 'mps'). 
                    If None, attempts to auto-detect.
        """
        self.model_name = model_name
        self.device = device or self._get_best_device() # Auto-detect if not specified
        logger.info(f"Initializing Reranker with model '{self.model_name}' on device '{self.device}'")
        
        try:
            # Load the CrossEncoder model
            self.model = CrossEncoder(self.model_name, device=self.device, max_length=512) # max_length default for many models
            logger.info(f"Successfully loaded model '{self.model_name}' to device '{self.device}'.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}", exc_info=True)
            # Fallback to CPU if specific device fails? Or just raise?
            # For now, let's raise to make the issue clear.
            raise RuntimeError(f"Could not load model '{self.model_name}'. Ensure it's installed or accessible.") from e

    def _get_best_device(self) -> str:
        """Auto-detects the best available device."""
        if torch.cuda.is_available():
            logger.info("CUDA detected. Using GPU.")
            return "cuda"
        # Check for Apple Silicon MPS (requires PyTorch >= 1.12)
        elif torch.backends.mps.is_available():
             if torch.backends.mps.is_built(): # Extra check for older PyTorch versions
                logger.info("MPS detected. Using Apple Silicon GPU.")
                return "mps"
             else:
                logger.warning("MPS available but not built. Falling back to CPU.")
                return "cpu"
        else:
            logger.info("No GPU detected (CUDA or MPS). Using CPU.")
            return "cpu"

    def _prepare_input_pairs(self, query: str, documents: List[Union[str, Dict[str, Any]]]) -> Tuple[List[Tuple[str, str]], List[int]]:
        """Prepares query-document pairs for the CrossEncoder model."""
        pairs = []
        original_indices = []
        for i, doc in enumerate(documents):
            doc_text = doc if isinstance(doc, str) else doc.get("text", "") # Handle string or dict
            if doc_text: # Avoid empty documents
                pairs.append((query, doc_text))
                original_indices.append(i)
            else:
                logger.warning(f"Skipping empty document at index {i}.")
        return pairs, original_indices

    def compute_scores(self, query: str, documents: List[Union[str, Dict[str, Any]]]) -> List[Tuple[int, float]]:
        """
        Computes relevance scores for a query and a list of documents.

        Args:
            query: The search query.
            documents: A list of documents (strings or dictionaries with a 'text' key).

        Returns:
            A list of tuples, where each tuple contains the original index 
            of the document and its relevance score.
        """
        if not documents:
            return []

        sentence_pairs, original_indices = self._prepare_input_pairs(query, documents)
        
        if not sentence_pairs:
            logger.warning("No valid document pairs found after preparation.")
            return []

        logger.info(f"Computing scores for {len(sentence_pairs)} query-document pairs...")
        # The CrossEncoder model's predict method handles batching internally
        # It expects a list of [query, passage] pairs
        scores = self.model.predict(sentence_pairs, show_progress_bar=False) # Progress bar can be noisy
        logger.info("Score computation finished.")

        if len(scores) != len(original_indices):
             logger.error("Mismatch between number of scores and original indices. This shouldn't happen.")
             # Handle error case, maybe return empty or raise? For now, return empty
             return []

        # Combine original indices with scores
        indexed_scores = list(zip(original_indices, scores))
        
        return indexed_scores
