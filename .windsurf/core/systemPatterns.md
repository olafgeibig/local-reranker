# System Patterns

*   **Architecture**: To be defined. Likely a simple service exposing an API endpoint, packaged for PyPI distribution. Needs to efficiently load and utilize ML models.
*   **Design Patterns**: To be defined. Considerations include efficient model loading/caching, request handling, and hardware-specific execution paths (MPS/CUDA/CPU).
*   **Core Components**: 
    *   API Endpoint Handler (e.g., FastAPI/Flask)
    *   Reranking Logic Module
    *   Model Loader/Manager
    *   Hardware Abstraction/Selector (for MPS/CUDA/CPU)
