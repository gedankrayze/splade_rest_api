"""
API routes for document management
"""

from fastapi import APIRouter, HTTPException, Path, status, Depends

from app.api.dependencies import get_splade_service
from app.core.splade_service import SpladeService
from app.models.schema import Document, DocumentBatch

router = APIRouter()


# Routes for adding documents to a collection
@router.post("/{collection_id}", status_code=status.HTTP_201_CREATED)
@router.post("/{collection_id}/", status_code=status.HTTP_201_CREATED)
async def add_document(collection_id: str, document: Document,
                       splade_service: SpladeService = Depends(get_splade_service)):
    """Add a document to a collection"""
    success = splade_service.add_document(collection_id, document)

    if not success:
        if not splade_service.get_collection(collection_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection with ID {collection_id} not found"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with ID {document.id} already exists in collection {collection_id}"
            )

    return {"status": "success", "message": f"Document {document.id} added to collection {collection_id}"}


# Routes for batch adding documents
@router.post("/{collection_id}/batch", status_code=status.HTTP_201_CREATED)
@router.post("/{collection_id}/batch/", status_code=status.HTTP_201_CREATED)
async def batch_add_documents(collection_id: str, batch: DocumentBatch,
                              splade_service: SpladeService = Depends(get_splade_service)):
    """Add multiple documents to a collection in batch"""
    if not splade_service.get_collection(collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found"
        )

    count = splade_service.batch_add_documents(collection_id, batch.documents)

    return {"status": "success", "message": f"{count} documents added to collection {collection_id}"}


# Routes for getting a document
@router.get("/{collection_id}/{document_id}", response_model=Document)
@router.get("/{collection_id}/{document_id}/", response_model=Document)
async def get_document(
        collection_id: str = Path(..., description="Collection ID"),
        document_id: str = Path(..., description="Document ID"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """Get a document from a collection"""
    document = splade_service.get_document(collection_id, document_id)

    if not document:
        if not splade_service.get_collection(collection_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection with ID {collection_id} not found"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found in collection {collection_id}"
            )

    return document


# Routes for deleting a document
@router.delete("/{collection_id}/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@router.delete("/{collection_id}/{document_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
        collection_id: str = Path(..., description="Collection ID"),
        document_id: str = Path(..., description="Document ID"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """Delete a document from a collection"""
    success = splade_service.remove_document(collection_id, document_id)

    if not success:
        if not splade_service.get_collection(collection_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection with ID {collection_id} not found"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found in collection {collection_id}"
            )
