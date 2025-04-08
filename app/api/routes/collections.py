"""
API routes for collection management
"""

from fastapi import APIRouter, HTTPException, status, Depends

from app.api.dependencies import get_splade_service
from app.core.splade_service import SpladeService
from app.models.schema import Collection, CollectionList, CollectionStats

router = APIRouter()


# Routes for collections list - both with and without trailing slash
@router.get("/", response_model=CollectionList)
@router.get("", response_model=CollectionList)
async def list_collections(splade_service: SpladeService = Depends(get_splade_service)):
    """List all collections"""
    collections = splade_service.list_collections()
    return {"collections": collections}


# Routes for getting collection details
@router.get("/{collection_id}", response_model=Collection)
@router.get("/{collection_id}/", response_model=Collection)
async def get_collection(collection_id: str, splade_service: SpladeService = Depends(get_splade_service)):
    """Get collection details"""
    collection = splade_service.get_collection(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found"
        )
    return collection


# Routes for getting collection stats
@router.get("/{collection_id}/stats", response_model=CollectionStats)
@router.get("/{collection_id}/stats/", response_model=CollectionStats)
async def get_collection_stats(collection_id: str, splade_service: SpladeService = Depends(get_splade_service)):
    """Get collection statistics"""
    collection = splade_service.get_collection(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found"
        )
    return collection["stats"]


# Routes for creating collections - both with and without trailing slash
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=Collection)
@router.post("", status_code=status.HTTP_201_CREATED, response_model=Collection)
async def create_collection(collection: Collection, splade_service: SpladeService = Depends(get_splade_service)):
    """Create a new collection"""
    success = splade_service.create_collection(
        collection_id=collection.id,
        name=collection.name,
        description=collection.description
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection with ID {collection.id} already exists"
        )

    return collection


# Routes for deleting collections
@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
@router.delete("/{collection_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_id: str, splade_service: SpladeService = Depends(get_splade_service)):
    """Delete a collection"""
    success = splade_service.delete_collection(collection_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found"
        )
