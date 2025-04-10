from fastapi import APIRouter, Depends, HTTPException, Response, status
from typing import Dict, List, Optional, Any

from src.auth import get_admin_user
from src.api.models import StatusResponse
from src.utils import get_search_engine

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_admin_user)]
)

@router.post("/reindex_embeddings", response_model=StatusResponse)
async def reindex_embeddings():
    """Admin endpoint to reindex all images with CLIP embeddings.
    
    This process:
    1. Fetches all existing metadata
    2. Ensures each document has a valid CLIP embedding
    3. Bulk reindexes all documents to Elasticsearch
    
    Returns:
        StatusResponse: Status of the reindexing operation
    """
    search_engine = get_search_engine()
    result = search_engine.reindex_all_with_embeddings()
    
    if result:
        return StatusResponse(success=True, message="All images have been reindexed with embeddings")
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reindex images with embeddings"
        ) 