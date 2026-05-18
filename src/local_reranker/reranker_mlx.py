# -*- coding: utf-8 -*-
"""MLX-based reranker implementation for Apple Silicon optimization."""

from typing import List, Dict, Any, Optional
import logging

from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult, RerankDocument
from .batch_manager import BatchManager
from .batch_processor import BatchProcessor, DocumentTextExtractor
from .config import get_mlx_platform_error

logger = logging.getLogger(__name__)


class Reranker(RerankerProtocol):
    """MLX implementation of reranker protocol for Apple Silicon optimization."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3-mlx",
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        disable_batching: bool = False,
    ):
        """Initialize MLX reranker with batching support.

        Args:
            model_name: The name of the MLX model to load.
            device: The device to run the model on. MLX auto-detects Apple Silicon GPU/CPU.
                    This parameter is kept for protocol compatibility but ignored.
            batch_size: Number of documents per batch (auto-detected if None).
            disable_batching: If True, process all documents in one batch.

        Raises:
            ImportError: If MLX dependencies are not installed.
            RuntimeError: If model loading fails.
        """
        platform_error = get_mlx_platform_error()
        if platform_error is not None:
            raise RuntimeError(
                f"{platform_error} Use '--backend pytorch' or set "
                "RERANKER_BACKEND_TYPE=pytorch. Install `local-reranker[mlx]` "
                "only on macOS Apple Silicon."
            )

        self.model_name = model_name
        self.device = device

        self.batch_manager = BatchManager(
            batch_size=batch_size, disable_batching=disable_batching
        )

        try:
            model_path = self._prepare_model_files(model_name)
            self.model = self._load_mlx_reranker(model_path)
            logger.info(f"Successfully loaded MLX reranker with batching: {model_name}")

        except (ImportError, OSError) as e:
            raise ImportError(
                "MLX backend requested, but the optional MLX dependencies could "
                "not be imported. This is expected on Linux containers because "
                "Apple's MLX runtime is unavailable there. Use '--backend pytorch' "
                "or install `local-reranker[mlx]` on macOS Apple Silicon."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model '{model_name}': {e}") from e

    def _prepare_model_files(self, model_name: str) -> str:
        """Prepare model files by downloading from HuggingFace if needed."""
        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
            )
            return model_path

        except ImportError:
            raise ImportError(
                "huggingface-hub not found. Install with: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {e}") from e

    def _load_mlx_reranker(self, model_path: str):
        """Load MLX reranker using JinaMLXReranker.

        Args:
            model_path: Path to the MLX model directory.

        Returns:
            JinaMLXReranker instance.

        Raises:
            RuntimeError: If model or projector loading fails.
        """
        try:
            from .jina_mlx_reranker import JinaMLXReranker
        except (ImportError, OSError) as exc:
            raise ImportError(
                "MLX backend requested, but the MLX runtime could not be loaded. "
                "On Docker/Linux this usually means the platform does not support "
                "MLX shared libraries such as libmlx.so. Use '--backend pytorch' "
                "or install `local-reranker[mlx]` on macOS Apple Silicon."
            ) from exc

        projector_path = f"{model_path}/projector.safetensors"

        return JinaMLXReranker(
            model_path=model_path,
            projector_path=projector_path,
        )

    def _prepare_inputs(self, documents: List[str]) -> List[str]:
        """Prepare documents for the MLX model.

        Args:
            documents: List of document texts

        Returns:
            List of document texts (as-is)
        """
        return documents

    def _run_inference(
        self, query: str, documents: List[str], return_documents: Optional[bool]
    ) -> List[Dict[str, Any]]:
        """Run inference on documents for a query.

        Args:
            query: The query text
            documents: List of document texts
            return_documents: Whether to include document content in results

        Returns:
            List of result dictionaries from the model
        """
        return self.model.rerank(
            query=query,
            documents=documents,
            top_n=None,
            return_embeddings=return_documents or False,
        )

    def _convert_batch_to_results(
        self,
        batch_results: List[Dict[str, Any]],
        original_indices: List[int],
        return_documents: Optional[bool],
        batch_docs: List[str],
    ) -> List[RerankResult]:
        """Convert model batch results to RerankResult objects.

        Args:
            batch_results: List of result dictionaries from the model
            original_indices: Original document indices
            return_documents: Whether to include document content in results
            batch_docs: List of document texts in the batch

        Returns:
            List of RerankResult objects
        """
        results = []
        for result_dict in batch_results:
            relevance_score = float(result_dict["relevance_score"])
            batch_relative_index = int(result_dict["index"])

            document = None
            if return_documents and batch_relative_index < len(batch_docs):
                doc_text = batch_docs[batch_relative_index]
                if doc_text:
                    document = RerankDocument(text=doc_text)

            results.append(
                RerankResult(
                    document=document,
                    index=original_indices[batch_relative_index],
                    relevance_score=relevance_score,
                )
            )
        return results

    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """Rerank documents using the MLX backend with batching support.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).

        Raises:
            ValueError: If the request is invalid.
            RuntimeError: If reranking fails.
        """
        if not request.query or not request.query.strip():
            logger.warning("[MLX] Empty query provided")
            return []

        if not request.documents:
            logger.info("[MLX] No documents to rerank")
            return []

        filtered_docs = [
            doc
            for doc in request.documents
            if doc and DocumentTextExtractor.extract(doc)
        ]

        if not filtered_docs:
            logger.info("[MLX] No valid documents after filtering")
            return []

        batches, batch_indices = self.batch_manager.create_batches(request)

        if not batches:
            logger.warning("[MLX] No batches created")
            return []

        all_batch_results = []
        for batch_idx, (batch_docs, original_indices) in enumerate(
            zip(batches, batch_indices)
        ):
            logger.info(
                f"[MLX] Processing batch {batch_idx + 1}/{len(batches)}: "
                f"{len(batch_docs)} documents"
            )

            inputs = self._prepare_inputs(batch_docs)
            raw_results = self._run_inference(
                request.query, inputs, request.return_documents or False
            )

            if len(raw_results) != len(batch_docs):
                logger.error(
                    f"[MLX] Result count mismatch in batch {batch_idx + 1}: "
                    f"expected {len(batch_docs)} results, got {len(raw_results)}"
                )
                return []

            batch_results = self._convert_batch_to_results(
                raw_results, original_indices, request.return_documents, batch_docs
            )
            all_batch_results.append(batch_results)

            logger.info(
                f"[MLX] Batch {batch_idx + 1} completed: {len(batch_results)} results"
            )

        return BatchProcessor.process_batched_results(all_batch_results, request.top_n)
