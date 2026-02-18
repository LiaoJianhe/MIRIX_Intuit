"""
Redis search provider for Mirix.

Wraps RedisMemoryClient to implement the search provider interface (vector,
text, recency). Registered alongside RedisCacheProvider when Redis is enabled.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mirix.log import get_logger

if TYPE_CHECKING:
    from mirix.database.redis_client import RedisMemoryClient

logger = get_logger(__name__)


class RedisSearchProvider:
    """
    Redis search provider implementation.

    Delegates to RedisMemoryClient for all search operations. Exposes the same
    index name constants so callers can use search_provider.EPISODIC_INDEX etc.
    """

    # Index names (must match RedisMemoryClient)
    BLOCK_INDEX = "idx:blocks"
    MESSAGE_INDEX = "idx:messages"
    EPISODIC_INDEX = "idx:episodic_memory"
    SEMANTIC_INDEX = "idx:semantic_memory"
    PROCEDURAL_INDEX = "idx:procedural_memory"
    RESOURCE_INDEX = "idx:resource_memory"
    KNOWLEDGE_INDEX = "idx:knowledge_vault"
    ORGANIZATION_INDEX = "idx:organizations"
    USER_INDEX = "idx:users"
    AGENT_INDEX = "idx:agents"
    TOOL_INDEX = "idx:tools"

    def __init__(self, redis_client: "RedisMemoryClient") -> None:
        self.redis_client = redis_client
        logger.info("Initialized RedisSearchProvider")

    async def search_recent(
        self,
        index_name: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        sort_by: str = "created_at_ts",
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Recency-sorted search; delegates to RedisMemoryClient."""
        return await self.redis_client.search_recent(
            index_name=index_name,
            limit=limit,
            user_id=user_id,
            organization_id=organization_id,
            sort_by=sort_by,
            return_fields=return_fields,
            filter_tags=filter_tags,
            start_date=start_date,
            end_date=end_date,
        )

    async def search_vector(
        self,
        index_name: str,
        embedding: List[float],
        vector_field: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search; delegates to RedisMemoryClient."""
        return await self.redis_client.search_vector(
            index_name=index_name,
            embedding=embedding,
            vector_field=vector_field,
            limit=limit,
            user_id=user_id,
            organization_id=organization_id,
            return_fields=return_fields,
            filter_tags=filter_tags,
            start_date=start_date,
            end_date=end_date,
        )

    async def search_text(
        self,
        index_name: str,
        query: str,
        search_fields: List[str],
        limit: int = 10,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search; delegates to RedisMemoryClient."""
        return await self.redis_client.search_text(
            index_name=index_name,
            query=query,
            search_fields=search_fields,
            limit=limit,
            user_id=user_id,
            organization_id=organization_id,
            return_fields=return_fields,
            filter_tags=filter_tags,
            start_date=start_date,
            end_date=end_date,
        )

    async def search_recent_by_org(
        self,
        index_name: str,
        limit: int = 10,
        organization_id: Optional[str] = None,
        sort_by: str = "created_at_ts",
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Recency search by org; delegates to RedisMemoryClient."""
        return await self.redis_client.search_recent_by_org(
            index_name=index_name,
            limit=limit,
            organization_id=organization_id,
            sort_by=sort_by,
            return_fields=return_fields,
            filter_tags=filter_tags,
            start_date=start_date,
            end_date=end_date,
        )

    async def search_vector_by_org(
        self,
        index_name: str,
        embedding: List[float],
        vector_field: str,
        limit: int = 10,
        organization_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Vector search by org; delegates to RedisMemoryClient."""
        return await self.redis_client.search_vector_by_org(
            index_name=index_name,
            embedding=embedding,
            vector_field=vector_field,
            limit=limit,
            organization_id=organization_id,
            return_fields=return_fields,
            filter_tags=filter_tags,
            start_date=start_date,
            end_date=end_date,
        )

    async def search_text_by_org(
        self,
        index_name: str,
        query_text: str,
        search_field: str,
        search_method: str,
        limit: int = 10,
        organization_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Text search by org; delegates to RedisMemoryClient."""
        return await self.redis_client.search_text_by_org(
            index_name=index_name,
            query_text=query_text,
            search_field=search_field,
            search_method=search_method,
            limit=limit,
            organization_id=organization_id,
            filter_tags=filter_tags,
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def clean_search_fields(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove search-backend-specific fields before Pydantic validation."""
        from mirix.database.redis_client import RedisMemoryClient

        return RedisMemoryClient.clean_redis_fields(items)
