"""
FastAPI dependencies
"""

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.splade_service import SpladeService


async def get_splade_service():
    """
    Dependency that ensures the SPLADE service is initialized.
    Returns the service if initialized, or initializes it if not.
    """
    global splade_service

    if splade_service is None:
        try:
            # Try to initialize the service
            splade_service = SpladeService()
        except Exception as e:
            # If it fails, raise an exception
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"SPLADE service is not available: {str(e)}. Please check if the model directory exists: {settings.MODEL_DIR}"
            )

    return splade_service
