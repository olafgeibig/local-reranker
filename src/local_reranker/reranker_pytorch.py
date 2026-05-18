# -*- coding: utf-8 -*-
"""PyTorch implementation of the reranker protocol."""

import logging
import warnings
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from sentence_transformers import CrossEncoder
from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult, RerankDocument
from .batch_manager import BatchManager
from .batch_processor import BatchProcessor, DocumentTextExtractor

# Suppress torch_dtype deprecation warning from Jina model files
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*Use `dtype` instead")

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"


class Reranker(RerankerProtocol):
    """PyTorch implementation of the reranker protocol using CrossEncoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        disable_batching: bool = False,
    ):
        """Initializes the Reranker with batching support.

        Args:
            model_name: The name of the CrossEncoder model to load.
            device: The device to run the model on ('cpu', 'cuda', 'mps').
            batch_size: Number of documents per batch (auto-detected if None).
            disable_batching: If True, process all documents in one batch.
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.batch_manager = BatchManager(
            batch_size=batch_size, disable_batching=disable_batching
        )

        logger.info(
            f"Initializing Reranker with model '{self.model_name}' on device '{self.device}' "
            f"with batching (batch_size={self.batch_manager.batch_size})"
        )

        try:
            logger.info(
                f"Attempting basic CrossEncoder initialization for '{self.model_name}'..."
            )
            self.model = CrossEncoder(
                model_name_or_path=self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info(
                f"Successfully loaded model '{self.model_name}' to device '{self.device}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to load model '{self.model_name}': {e}", exc_info=True
            )
            raise RuntimeError(
                f"Could not load model '{self.model_name}'. Ensure it's installed or accessible."
            ) from e

    def _get_best_device(self) -> str:
        """Auto-detects the best available device."""
        if torch.cuda.is_available():
            logger.info("CUDA detected. Using GPU.")
            return "cuda"
        elif torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("MPS detected. Using Apple Silicon GPU.")
                return "mps"
            else:
                logger.warning("MPS available but not built. Falling back to CPU.")
                return "cpu"
        else:
            logger.info("No GPU detected (CUDA or MPS). Using CPU.")
            return "cpu"

    def _prepare_inputs(
        self, query: str, documents: List[str]
    ) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for the CrossEncoder model.

        Args:
            query: The query text
            documents: List of document texts

        Returns:
            List of query-document tuples
        """
        return [(query, doc) for doc in documents]

    def _run_inference(
        self, inputs: List[Tuple[str, str]]
    ) -> Union[List[float], "np.ndarray"]:
        """Run inference on query-document pairs.

        Args:
            inputs: List of query-document tuples

        Returns:
            List or array of relevance scores
        """
        return self.model.predict(inputs, show_progress_bar=False)

    def _convert_batch_to_results(
        self,
        scores: List[float],
        original_indices: List[int],
        return_documents: Optional[bool],
        batch_docs: List[str],
    ) -> List[RerankResult]:
        """Convert model scores to RerankResult objects.

        Args:
            scores: List of relevance scores from the model
            original_indices: Original document indices
            return_documents: Whether to include document content in results
            batch_docs: List of document texts in the batch

        Returns:
            List of RerankResult objects
        """
        results = []
        for idx, (score, original_idx) in enumerate(zip(scores, original_indices)):
            document = None
            if return_documents and idx < len(batch_docs):
                document = RerankDocument(text=batch_docs[idx])

            results.append(
                RerankResult(
                    document=document,
                    index=original_idx,
                    relevance_score=float(score),
                )
            )
        return results

    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """Reranks documents based on the request using batching.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        if not request.documents:
            return []

        filtered_docs = [
            doc
            for doc in request.documents
            if doc and DocumentTextExtractor.extract(doc)
        ]

        if not filtered_docs:
            return []

        batches, batch_indices = self.batch_manager.create_batches(request)

        if not batches:
            return []

        all_batch_results = []
        for batch_docs, original_indices in zip(batches, batch_indices):
            inputs = self._prepare_inputs(request.query, batch_docs)
            scores = self._run_inference(inputs)

            if len(scores) != len(batch_docs):
                logger.error(
                    f"[PyTorch] Result count mismatch: expected {len(batch_docs)} results, "
                    f"got {len(scores)}"
                )
                return []

            batch_results = self._convert_batch_to_results(
                list(scores), original_indices, request.return_documents, batch_docs
            )
            all_batch_results.append(batch_results)

        return BatchProcessor.process_batched_results(all_batch_results, request.top_n)
