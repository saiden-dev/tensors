"""FastAPI route handlers for unified model search across providers."""

from __future__ import annotations

import contextlib
import logging
from enum import StrEnum
from typing import Annotated, Any

from fastapi import APIRouter, Query

from tensors.api import search_civitai
from tensors.config import (
    BaseModel as BaseModelEnum,
)
from tensors.config import (
    CommercialUse as CommercialUseEnum,
)
from tensors.config import (
    ModelType,
    load_api_key,
)
from tensors.config import (
    NsfwLevel as NsfwLevelEnum,
)
from tensors.config import (
    Period as PeriodEnum,
)
from tensors.config import (
    SortOrder as SortOrderEnum,
)
from tensors.db import Database
from tensors.hf import search_hf_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["Search"])


class Provider(StrEnum):
    """Search provider options."""

    civitai = "civitai"
    hf = "hf"
    all = "all"


class SortOrder(StrEnum):
    """Sort order options."""

    downloads = "downloads"
    rating = "rating"
    newest = "newest"


@router.get("")
async def search_models(
    query: Annotated[str | None, Query(description="Search query")] = None,
    provider: Annotated[Provider, Query(description="Search provider (civitai, hf, or all)")] = Provider.all,
    # CivitAI-specific
    types: Annotated[str | None, Query(description="Model type - CivitAI (Checkpoint, LORA, etc.)")] = None,
    base_models: Annotated[str | None, Query(alias="baseModels", description="Base model - CivitAI")] = None,
    period: Annotated[str | None, Query(description="Time period - CivitAI (AllTime, Year, Month, Week, Day)")] = None,
    nsfw: Annotated[str | None, Query(description="NSFW level - CivitAI (None, Soft, Mature, X)")] = None,
    sfw: Annotated[bool, Query(description="Exclude NSFW - CivitAI")] = False,
    commercial: Annotated[str | None, Query(description="Commercial use - CivitAI (None, Image, Rent, Sell)")] = None,
    page: Annotated[int | None, Query(ge=1, description="Page number - CivitAI")] = None,
    # HuggingFace-specific
    pipeline: Annotated[str | None, Query(description="Pipeline tag - HuggingFace (text-to-image, etc.)")] = None,
    # Common
    sort: Annotated[SortOrder, Query(description="Sort order")] = SortOrder.downloads,
    limit: Annotated[int, Query(le=100, description="Max results per provider")] = 25,
    tag: Annotated[str | None, Query(description="Filter by tag")] = None,
    author: Annotated[str | None, Query(description="Filter by author/creator")] = None,
) -> dict[str, Any]:
    """Search models across CivitAI and/or Hugging Face.

    Returns results from selected provider(s). When provider=all, returns
    results from both CivitAI and Hugging Face in separate keys.
    """
    api_key = load_api_key()
    results: dict[str, Any] = {}

    # Search CivitAI
    if provider in (Provider.civitai, Provider.all):
        # Map sort order to CivitAI enum
        civitai_sort = SortOrderEnum.downloads
        if sort == SortOrder.rating:
            civitai_sort = SortOrderEnum.rating
        elif sort == SortOrder.newest:
            civitai_sort = SortOrderEnum.newest

        # Map other enums
        model_type = ModelType(types.lower()) if types else None
        base_model = None
        if base_models:
            with contextlib.suppress(ValueError):
                base_model = BaseModelEnum(base_models.lower())

        period_enum = None
        if period:
            with contextlib.suppress(ValueError):
                period_enum = PeriodEnum(period.lower())

        nsfw_filter: NsfwLevelEnum | bool | None = None
        if sfw:
            nsfw_filter = NsfwLevelEnum.none
        elif nsfw:
            with contextlib.suppress(ValueError):
                nsfw_filter = NsfwLevelEnum(nsfw.lower())

        commercial_enum = None
        if commercial:
            with contextlib.suppress(ValueError):
                commercial_enum = CommercialUseEnum(commercial.lower())

        civitai_results = search_civitai(
            query=query,
            model_type=model_type,
            base_model=base_model,
            sort=civitai_sort,
            limit=limit,
            api_key=api_key,
            period=period_enum,
            nsfw=nsfw_filter,
            tag=tag,
            username=author,
            page=page,
            commercial_use=commercial_enum,
        )

        if civitai_results:
            results["civitai"] = civitai_results

    # Search Hugging Face
    if provider in (Provider.hf, Provider.all):
        hf_sort = "downloads"
        if sort == SortOrder.rating:
            hf_sort = "likes"
        elif sort == SortOrder.newest:
            hf_sort = "created_at"

        tags = [tag] if tag else None
        hf_results = search_hf_models(
            query=query,
            author=author,
            tags=tags,
            pipeline_tag=pipeline,
            sort=hf_sort,
            limit=limit,
        )

        if hf_results:
            results["huggingface"] = hf_results
            # Cache HF models
            try:
                with Database() as db:
                    db.init_schema()
                    for model_data in hf_results:
                        db.cache_hf_model(model_data)
            except Exception as e:
                logger.warning("Failed to cache HF search results: %s", e)

    return results


def create_search_router() -> APIRouter:
    """Return the unified search API router."""
    return router
