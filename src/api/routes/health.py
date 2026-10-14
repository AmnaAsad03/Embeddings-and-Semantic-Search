from fastapi import APIRouter


router = APIRouter(
    tags=["Health"]
)


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "medical-knowledge-assistant",
        "version": "0.1.0",
    }