from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Medical search query",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve",
    )


class SearchResult(BaseModel):
    rank: int
    content: str
    metadata: dict
    vector_score: float
    bm25_score: float
    hybrid_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    result_count: int


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Medical question",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of evidence sources to use",
    )


class Citation(BaseModel):
    citation_id: int
    document_id: str | None = None
    title: str | None = None
    source: str | None = None
    source_url: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    category: str
    action: str
    citations: list[Citation]
    evidence_sufficient: bool
    safety_category: str
    safety_allowed: bool
    safety_mode: str
    safety_severity: str