from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.search import router as search_router
from src.api.routes.ask import router as ask_router


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "Medical Knowledge Assistant"

APP_DESCRIPTION = (
    "Safe, evidence-grounded medical information system"
)

APP_VERSION = "0.1.0"


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


# ============================================================
# REGISTER ROUTES
# ============================================================

app.include_router(health_router)
app.include_router(search_router)
app.include_router(ask_router)