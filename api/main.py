"""FastAPI REST Application Entrypoint for Corpus Mind (Improvement 32)."""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db

from api.routers import auth as auth_router
from api.routers import users as users_router
from api.routers import documents as docs_router
from api.routers import rag as rag_router
from api.routers import audit as audit_router

logger = logging.getLogger("api")


# Ensure default admin account exists on initialization
try:
    AuthManager.initialize_admin_account()
except Exception as e:
    logger.warning(f"Could not initialize admin account on startup: {e}")

app = FastAPI(
    title="Corpus Mind Enterprise RAG API",
    description=(
        "Production-grade RESTful API for Corpus Mind: document intelligence, "
        "hybrid vector search, multi-provider LLM orchestration, fine-grained "
        "document permissions, and security audit logs."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Middleware
origins = getattr(config, "CORS_ORIGINS", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(docs_router.router)
app.include_router(rag_router.router)
app.include_router(audit_router.router)


@app.get("/", tags=["Health & Status"])
async def root():
    """Root metadata endpoint."""
    return {
        "service": "Corpus Mind Enterprise RAG API",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "database_engine": "PostgreSQL" if db.is_postgres else "SQLite"
    }


@app.get("/health", tags=["Health & Status"])
async def health_check():
    """Health check for Docker / Kubernetes container liveness probes."""
    db_healthy = False
    try:
        res = db._execute("SELECT 1 as alive", fetch_one=True)
        db_healthy = (res and res.get("alive") == 1)
    except Exception as e:
        logger.error(f"Health check DB probe failed: {e}")

    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": {
            "status": "connected" if db_healthy else "error",
            "engine": "PostgreSQL" if db.is_postgres else "SQLite"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=getattr(config, "API_HOST", "0.0.0.0"),
        port=getattr(config, "API_PORT", 8000),
        reload=True
    )
