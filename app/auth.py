from dataclasses import dataclass

from fastapi import Depends, Header, Request

from app.config import Settings
from app.errors import ApiError

CLEARANCE = {
    "analyst": {"public", "internal"},
    "admin": {"public", "internal", "confidential"},
}


@dataclass(frozen=True)
class Principal:
    tenant: str
    role: str

    @property
    def clearance(self) -> set[str]:
        return CLEARANCE[self.role]


def get_principal(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
    """Identity comes ONLY from the API key. Nothing in a body, query string or document can change it."""
    if not x_api_key:
        raise ApiError(401, "missing_api_key", "X-API-Key header is required")
    settings: Settings = request.app.state.settings
    entry = settings.api_keys.get(x_api_key)
    if entry is None:
        raise ApiError(401, "invalid_api_key", "Unknown API key")
    return Principal(tenant=entry.tenant, role=entry.role)


def require_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if principal.role != "admin":
        raise ApiError(403, "forbidden", "This operation requires the admin role")
    return principal
