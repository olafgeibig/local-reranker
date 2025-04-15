# Product Context

*   **Target Audience**: Developers implementing RAG applications who need a local, private, and performant reranking solution, especially on Mac hardware.
*   **User Needs**: 
    *   Privacy through local data processing.
    *   A simple, drop-in replacement for cloud-based reranker APIs (initially Jina).
    *   Optimized performance on specific hardware (Mac ARM).
    *   Easy integration via PyPI packaging.
    *   Low latency for interactive use cases.
*   **Key Features**: 
    *   Local implementation of the Jina `/v1/rerank` endpoint.
    *   Compatibility with `jina-reranker-v2-base-multilingual` model.
    *   Support for key API parameters (`model`, `query`, `documents`, `top_n`, `return_documents`).
    *   Hardware acceleration (Mac MPS, NVIDIA CUDA) with CPU fallback.
    *   Performance targeting interactive RAG on Mac M1 (16GB RAM).
