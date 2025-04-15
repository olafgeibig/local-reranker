# -*- coding: utf-8 -*-
"""FastAPI application for the local reranker service."""

import logging
import time
import uuid
from typing import List
from contextlib import asynccontextmanager
import torch

from fastapi import FastAPI, HTTPException, Depends, Request 
from .models import RerankRequest, RerankResponse, RerankResult, RerankDocument
from .reranker import Reranker, DEFAULT_MODEL_NAME 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Lifespan Management (Load model on startup, cleanup on shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the reranker model's lifecycle."""
    logger.info("Lifespan startup: Loading reranker model...")
    reranker_instance = None
    try:
        # TODO: Make model name configurable (e.g., via environment variable)
        reranker_instance = Reranker(model_name=DEFAULT_MODEL_NAME)
        app.state.reranker = reranker_instance # Store instance in app state
        logger.info("Reranker model loaded successfully and stored in app state.")
    except Exception as e:
        logger.error(f"Fatal error: Could not load reranker model during startup: {e}", exc_info=True)
        app.state.reranker = None # Ensure it's None if loading failed
    
    yield # Application runs here
    
    # --- Cleanup logic ---    
    logger.info("Lifespan shutdown: Releasing resources...")
    current_reranker = getattr(app.state, 'reranker', None)
    if current_reranker and hasattr(current_reranker.model, 'cpu'): # Basic check if it's a torch model
        try:
            # Ensure model and tensors are moved to CPU before deletion if applicable
            # Note: sentence-transformers CrossEncoder might not need explicit deletion
            # but clearing cache is good practice.
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Clear CUDA cache if applicable
            logger.info("Model resources released.")
        except Exception as e:
            logger.error(f"Error during model resource cleanup: {e}", exc_info=True)
    app.state.reranker = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Local Reranker API",
    description="Provides a local implementation of reranker APIs (starting with Jina).",
    version="0.0.1", 
    lifespan=lifespan 
)

# --- Dependency Injection for Reranker --- 
def get_reranker(request: Request): 
    # Retrieve the instance from app.state managed by lifespan
    reranker = getattr(request.app.state, 'reranker', None)
    if reranker is None:
        logger.error("Reranker instance is not available via app.state. Model loading might have failed during startup.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Reranker model not loaded.")
    return reranker

# --- API Endpoints ---
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_endpoint(
    request_body: RerankRequest, 
    reranker: Reranker = Depends(get_reranker) 
):
    """Handles reranking requests, compatible with Jina's /v1/rerank API."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Received rerank request.")
    logger.info(f"[{request_id}] Reranking query: {request_body.query}")
    # logger.info(f"[{request_id}] Reranking request: {request_body}")
    try:
        # Call the reranker's rerank method
        # 1. Compute scores
        indexed_scores = reranker.compute_scores(
            query=request_body.query,
            documents=request_body.documents,
        )
        
        # 2. Sort results by score (descending)
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Apply top_n limit
        if request_body.top_n is not None:
            indexed_scores = indexed_scores[:request_body.top_n]

        # 4. Format the response
        results: List[RerankResult] = []
        for index, score in indexed_scores:
            doc_content = None
            if request_body.return_documents:
                original_doc = request_body.documents[index]
                doc_text = original_doc if isinstance(original_doc, str) else original_doc.get("text", "")
                doc_content = RerankDocument(text=doc_text)

            results.append(
                RerankResult(
                    document=doc_content,
                    index=index,
                    relevance_score=float(score) 
                )
            )
        # Add top score and first few characters of top document to the log message
        top_doc_preview = ""
        if results and request_body.return_documents and results[0].document:
            top_doc_preview = results[0].document.text[:50]
        logger.info(f"[{request_id}] Reranking done, top score: {results[0].relevance_score}, preview: {top_doc_preview}")
        # logger.info(f"results: {results}")
        response = RerankResponse(id=request_id, results=results)
        
        end_time = time.time()
        logger.debug(f"[{request_id}] Rerank request processed in {end_time - start_time:.4f} seconds.")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] Error processing rerank request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during reranking.")

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

# --- Main block for running with uvicorn directly ---
# For production, use: uvicorn local_reranker.api:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("local_reranker.api:app", host="127.0.0.1", port=8000, reload=True)
