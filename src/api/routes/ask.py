from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    AskRequest,
    AskResponse,
)

from src.api.errors import (
    BadRequestError,
    ServiceError,
)

from src.generation.rag_pipeline import MedicalRAGPipeline


router = APIRouter(
    tags=["Ask"]
)


pipeline = MedicalRAGPipeline()


@router.post(
    "/ask",
)
def ask(request: AskRequest):

    try:

        result = pipeline.process(
            question=request.question,
            top_k=request.top_k,
        )

    except ValueError as exc:

        raise BadRequestError(
            detail=str(exc)
        )

    except Exception:

        raise ServiceError(
            detail="Medical assistant service is temporarily unavailable."
        )

    def generate():

        answer = result["answer"]

        chunk_size = 50

        for index in range(
            0,
            len(answer),
            chunk_size,
        ):

            yield answer[
                index:index + chunk_size
            ]

    return StreamingResponse(
        generate(),
        media_type="text/plain",
    )