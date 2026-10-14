from fastapi import APIRouter

from src.api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)

from src.api.errors import (
    BadRequestError,
    ServiceError,
)

from src.retrieval.hybrid_search import HybridSearcher


router = APIRouter(
    tags=["Search"]
)


searcher = HybridSearcher()


@router.post(
    "/search",
    response_model=SearchResponse,
)
def search(request: SearchRequest):

    try:

        results = searcher.search(
            query=request.query,
            top_k=request.top_k,
        )

    except ValueError as exc:

        raise BadRequestError(
            detail=str(exc)
        )

    except Exception:

        raise ServiceError(
            detail="Search service is temporarily unavailable."
        )

    formatted_results = [
        SearchResult(
            rank=result["rank"],
            content=result["content"],
            metadata=result["metadata"],
            vector_score=result["vector_score"],
            bm25_score=result["bm25_score"],
            hybrid_score=result["hybrid_score"],
        )
        for result in results
    ]

    return SearchResponse(
        query=request.query,
        results=formatted_results,
        result_count=len(formatted_results),
    )