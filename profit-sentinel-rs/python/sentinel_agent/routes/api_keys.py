"""Enterprise API key management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..api_keys import (
    ApiTier,
    create_api_key,
    get_key_usage,
    list_api_keys,
    revoke_api_key,
)
from ..api_models import (
    ApiKeyListResponse,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
)
from ..dual_auth import UserContext


def create_api_keys_router(require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["api-keys"])

    @router.post(
        "/api-keys",
        response_model=CreateApiKeyResponse,
        dependencies=[Depends(require_auth)],
    )
    async def create_key(
        body: CreateApiKeyRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> CreateApiKeyResponse:
        """Create a new API key. The plaintext key is returned only once."""
        try:
            tier = ApiTier(body.tier)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier '{body.tier}'. Use: free, pro, enterprise",
            )
        plaintext, record = create_api_key(
            ctx.user_id, tier=tier, name=body.name, test=body.test
        )
        return CreateApiKeyResponse(key=plaintext, record=record.to_dict())

    @router.get(
        "/api-keys",
        response_model=ApiKeyListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_api_keys(
        ctx: UserContext = Depends(require_auth),
    ) -> ApiKeyListResponse:
        """List all API keys for the current user."""
        keys = list_api_keys(ctx.user_id)
        return ApiKeyListResponse(
            keys=[k.to_dict() for k in keys],
            total=len(keys),
        )

    @router.delete(
        "/api-keys/{key_id}",
        dependencies=[Depends(require_auth)],
    )
    async def revoke_key(
        key_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Revoke an API key."""
        revoked = revoke_api_key(key_id, ctx.user_id)
        if not revoked:
            raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
        return {"message": "API key revoked", "key_id": key_id}

    @router.get(
        "/api-keys/{key_id}/usage",
        dependencies=[Depends(require_auth)],
    )
    async def get_key_usage_stats(
        key_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Get usage statistics for an API key."""
        stats = get_key_usage(key_id, ctx.user_id)
        if not stats:
            raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
        return stats

    return router
