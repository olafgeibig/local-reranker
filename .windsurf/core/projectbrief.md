# Project Brief

*   **Project Name**: local-reranker
*   **Goal**: Provide a lightweight, high-performance, privacy-focused local utility application that mimics reranker API endpoints (starting with Jina's `/v1/rerank`) for use in RAG applications.
*   **Scope**: 
    *   Implement Jina `/v1/rerank` API compatibility locally.
    *   Support `jina-reranker-v2-base-multilingual` model.
    *   Optimize for Mac ARM (MPS) and support NVIDIA (CUDA) GPUs, with CPU fallback.
    *   Distribute via PyPI package.
    *   Design for future expansion to other APIs/models.
