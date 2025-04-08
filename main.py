#!/usr/bin/env python3
"""
SPLADE Content Server with FAISS
Main application entry point
"""

import uvicorn

from app.api.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
