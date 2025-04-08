"""
API middleware
"""

import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("middleware")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and handle errors globally
    """

    async def dispatch(self, request: Request, call_next):
        try:
            # Try to process the request
            return await call_next(request)
        except Exception as e:
            # Log the error
            logger.error(f"Unhandled exception: {str(e)}")

            # Return a 500 Internal Server Error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error. Please check if the SPLADE model is correctly loaded.",
                    "error": str(e)
                }
            )
