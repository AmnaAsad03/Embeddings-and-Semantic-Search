from fastapi import HTTPException


class APIError(HTTPException):
    """Base exception for API errors."""

    def __init__(
        self,
        status_code: int,
        detail: str,
    ):
        super().__init__(
            status_code=status_code,
            detail=detail,
        )


class BadRequestError(APIError):
    """Raised when the client sends an invalid request."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=400,
            detail=detail,
        )


class ServiceError(APIError):
    """Raised when an internal service operation fails."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=500,
            detail=detail,
        )