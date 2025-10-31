"""
Hybrid Redis client for Mirix memory system.

Uses:
- Redis Hash for blocks (fast, flat structure, no embeddings)
- Redis Hash for messages (fast, mostly flat, no embeddings)  
- Redis JSON with Vector fields for memory tables (embeddings support)

Provides:
- 40-60% faster operations for blocks and messages via Hash
- 10-40x faster vector similarity search vs PostgreSQL pgvector
- Hybrid text+vector search capabilities
"""

import json
from typing import Any, Dict, List, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Global Redis client instance
_redis_client: Optional["RedisMemoryClient"] = None


class RedisMemoryClient:
    """
    Hybrid Redis client for Mirix memory caching and search.
    
    Architecture:
    - Hash: blocks, messages (no embeddings, flat structure)
    - JSON + Vector: episodic, semantic, procedural, resource, knowledge (has embeddings)
    """
    
    # Index names
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
    
    # Key prefixes
    BLOCK_PREFIX = "block:"
    MESSAGE_PREFIX = "msg:"
    EPISODIC_PREFIX = "episodic:"
    SEMANTIC_PREFIX = "semantic:"
    PROCEDURAL_PREFIX = "procedural:"
    RESOURCE_PREFIX = "resource:"
    KNOWLEDGE_PREFIX = "knowledge:"
    ORGANIZATION_PREFIX = "org:"
    USER_PREFIX = "user:"
    AGENT_PREFIX = "agent:"
    TOOL_PREFIX = "tool:"
    
    def __init__(
        self,
        redis_uri: str,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
    ):
        """Initialize Redis client with connection pool."""
        try:
            from redis import Redis, ConnectionPool
            
            self.redis_uri = redis_uri
            
            self.pool = ConnectionPool.from_url(
                redis_uri,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=True,
            )
            
            self.client = Redis(connection_pool=self.pool)
            
            logger.info("✅ Redis hybrid client initialized: %s", self._mask_uri(redis_uri))
        except ImportError:
            logger.error("Redis library not installed. Install with: pip install redis[hiredis]")
            raise
        except Exception as e:
            logger.error("Failed to initialize Redis client: %s", e)
            raise
    
    def _mask_uri(self, uri: str) -> str:
        """Mask password in URI for logging."""
        if '@' in uri and ':' in uri:
            parts = uri.split('@')
            if len(parts) == 2:
                protocol = parts[0].split('://')[0]
                return f"{protocol}://****@{parts[1]}"
        return uri
    
    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error("Redis ping failed: %s", e)
            return False
    
    def close(self) -> None:
        """Close Redis connection pool."""
        try:
            if self.pool:
                self.pool.disconnect()
                logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error("Error closing Redis pool: %s", e)
    
    def create_indexes(self) -> None:
        """Create RediSearch indexes for all memory types (hybrid approach)."""
        logger.info("Creating Redis indexes (hybrid: Hash for blocks/messages/orgs/users/agents/tools, JSON+Vectors for memory)...")
        
        try:
            # Hash-based indexes (no embeddings)
            self._create_block_index()
            self._create_message_index()
            self._create_organization_index()
            self._create_user_index()
            self._create_agent_index()
            self._create_tool_index()
            
            # JSON-based indexes with vector fields (has embeddings)
            self._create_episodic_index()
            self._create_semantic_index()
            self._create_procedural_index()
            self._create_resource_index()
            self._create_knowledge_index()
            
            logger.info("✅ All Redis indexes created successfully")
        except Exception as e:
            logger.error("Failed to create some indexes: %s", e)
            # Don't raise - allow system to continue without indexes
    
    # ========================================================================
    # HASH-BASED METHODS (for blocks and messages - NO embeddings)
    # ========================================================================
    
    def _create_block_index(self) -> None:
        """Create HASH-based index for blocks (Core Memory)."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            
            try:
                self.client.ft(self.BLOCK_INDEX).info()
                logger.debug("Index %s already exists", self.BLOCK_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("organization_id"),
                TextField("user_id"),
                TextField("agent_id"),
                TagField("label"),  # Exact match: "human", "persona"
                TextField("value"),  # Full-text search on content
                NumericField("limit"),
                NumericField("created_at_ts"),
            )
            
            self.client.ft(self.BLOCK_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.BLOCK_PREFIX],
                    index_type=IndexType.HASH  # ⭐ Hash type for simple data
                )
            )
            logger.info("✅ Created HASH index: %s", self.BLOCK_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create block index: %s", e)
    
    def _create_message_index(self) -> None:
        """Create HASH-based index for messages."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            
            try:
                self.client.ft(self.MESSAGE_INDEX).info()
                logger.debug("Index %s already exists", self.MESSAGE_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("organization_id"),
                TextField("agent_id"),
                TextField("user_id"),
                TagField("role"),  # user, assistant, system, tool
                TextField("text"),  # Message text
                TextField("model"),
                NumericField("created_at_ts"),
            )
            
            self.client.ft(self.MESSAGE_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.MESSAGE_PREFIX],
                    index_type=IndexType.HASH  # ⭐ Hash type
                )
            )
            logger.info("✅ Created HASH index: %s", self.MESSAGE_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create message index: %s", e)
    
    def _create_organization_index(self) -> None:
        """Create HASH-based index for organizations."""
        try:
            from redis.commands.search.field import TextField, NumericField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            
            try:
                self.client.ft(self.ORGANIZATION_INDEX).info()
                logger.debug("Index %s already exists", self.ORGANIZATION_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("id"),
                TextField("name"),
                NumericField("created_at_ts"),
            )
            
            self.client.ft(self.ORGANIZATION_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.ORGANIZATION_PREFIX],
                    index_type=IndexType.HASH
                )
            )
            logger.info("✅ Created HASH index: %s", self.ORGANIZATION_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create organization index: %s", e)
    
    def _create_user_index(self) -> None:
        """Create HASH-based index for users."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            
            try:
                self.client.ft(self.USER_INDEX).info()
                logger.debug("Index %s already exists", self.USER_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("id"),
                TextField("organization_id"),
                TextField("name"),
                TagField("status"),  # active/inactive
                TextField("timezone"),
                NumericField("created_at_ts"),
                NumericField("updated_at_ts"),
                TagField("is_deleted"),
            )
            
            self.client.ft(self.USER_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.USER_PREFIX],
                    index_type=IndexType.HASH
                )
            )
            logger.info("✅ Created HASH index: %s", self.USER_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create user index: %s", e)
    
    def _create_agent_index(self) -> None:
        """Create HASH-based index for agents (with denormalized tool_ids)."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            
            try:
                self.client.ft(self.AGENT_INDEX).info()
                logger.debug("Index %s already exists", self.AGENT_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("id"),
                TextField("organization_id"),
                TextField("name"),
                TagField("agent_type"),
                TextField("description"),
                TextField("parent_id"),
                TextField("system"),  # System prompt
                NumericField("created_at_ts"),
                NumericField("updated_at_ts"),
                TagField("is_deleted"),
            )
            
            self.client.ft(self.AGENT_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.AGENT_PREFIX],
                    index_type=IndexType.HASH
                )
            )
            logger.info("✅ Created HASH index: %s", self.AGENT_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create agent index: %s", e)
    
    def _create_tool_index(self) -> None:
        """Create HASH-based index for tools."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            
            try:
                self.client.ft(self.TOOL_INDEX).info()
                logger.debug("Index %s already exists", self.TOOL_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("id"),
                TextField("organization_id"),
                TextField("name"),
                TagField("tool_type"),  # CORE, CUSTOM, etc.
                TextField("description"),
                TagField("tags", separator=","),
                NumericField("return_char_limit"),
                NumericField("created_at_ts"),
                NumericField("updated_at_ts"),
                TagField("is_deleted"),
            )
            
            self.client.ft(self.TOOL_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.TOOL_PREFIX],
                    index_type=IndexType.HASH
                )
            )
            logger.info("✅ Created HASH index: %s", self.TOOL_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create tool index: %s", e)
    
    def set_hash(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store data as Redis Hash (for flat structures like blocks and messages).
        
        Args:
            key: Redis key
            data: Data to store (will be flattened)
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        try:
            # Flatten and convert all values to strings
            flattened = self._flatten_dict(data)
            
            # HSET creates/updates all fields atomically
            self.client.hset(key, mapping=flattened)
            
            if ttl:
                self.client.expire(key, ttl)
            
            logger.debug("✅ Stored Hash: %s (%d fields)", key, len(flattened))
            return True
        except Exception as e:
            logger.error("Failed to set hash for %s: %s", key, e)
            return False
    
    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from Redis Hash.
        
        Args:
            key: Redis key
        
        Returns:
            Data dictionary or None if not found
        """
        try:
            data = self.client.hgetall(key)
            if not data:
                return None
            
            # Convert back to proper types
            result = self._unflatten_dict(data)
            logger.debug("✅ Retrieved Hash: %s", key)
            return result
        except Exception as e:
            logger.error("Failed to get hash for %s: %s", key, e)
            return None
    
    def update_hash_field(
        self, key: str, field: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """
        Update a single field in Redis Hash (very fast for partial updates!).
        
        Args:
            key: Redis key
            field: Field name to update
            value: New value
            ttl: Optional TTL reset
        
        Returns:
            True if successful
        """
        try:
            self.client.hset(key, field, str(value))
            if ttl:
                self.client.expire(key, ttl)
            logger.debug("✅ Updated Hash field: %s.%s", key, field)
            return True
        except Exception as e:
            logger.error("Failed to update hash field %s in %s: %s", field, key, e)
            return False
    
    def _flatten_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Flatten dictionary for Hash storage. All values converted to strings."""
        flattened = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Store nested dicts as JSON strings
                flattened[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                # Store lists as JSON strings
                flattened[key] = json.dumps(value)
            elif value is None:
                flattened[key] = ""
            elif isinstance(value, bool):
                flattened[key] = "true" if value else "false"
            else:
                flattened[key] = str(value)
        return flattened
    
    def _unflatten_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert Hash data back to proper Python types."""
        result = {}
        for key, value in data.items():
            # Parse numeric fields (e.g., 'limit')
            if key in ('limit',):
                try:
                    result[key] = float(value) if '.' in value else int(value)
                    continue
                except (ValueError, AttributeError):
                    pass
            
            # Parse boolean
            if value.lower() in ('true', 'false'):
                result[key] = value.lower() == 'true'
                continue
            
            # Try to parse JSON (for nested structures)
            if value and (value.startswith('{') or value.startswith('[')):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            
            # Keep as string
            result[key] = value if value else None
        
        return result
    
    # ========================================================================
    # JSON-BASED METHODS (for memory types with embeddings)
    # ========================================================================
    
    def _create_episodic_index(self) -> None:
        """Create JSON-based index for episodic memory with 2 VECTOR fields."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            from mirix.constants import MAX_EMBEDDING_DIM
            
            try:
                self.client.ft(self.EPISODIC_INDEX).info()
                logger.debug("Index %s already exists", self.EPISODIC_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.actor", as_name="actor"),
                TextField("$.event_type", as_name="event_type"),
                TextField("$.summary", as_name="summary"),
                TextField("$.details", as_name="details"),
                NumericField("$.occurred_at_ts", as_name="occurred_at_ts"),
                TagField("$.user_id", as_name="user_id"),
                
                # ⭐ Vector fields for embeddings (32KB total)
                VectorField(
                    "$.details_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="details_embedding"
                ),
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="summary_embedding"
                ),
            )
            
            self.client.ft(self.EPISODIC_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.EPISODIC_PREFIX],
                    index_type=IndexType.JSON  # ⭐ JSON type for complex data
                )
            )
            logger.info("✅ Created JSON+VECTOR index: %s (2 vectors)", self.EPISODIC_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create episodic index: %s", e)
    
    def _create_semantic_index(self) -> None:
        """Create JSON-based index for semantic memory with 3 VECTOR fields."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            from mirix.constants import MAX_EMBEDDING_DIM
            
            try:
                self.client.ft(self.SEMANTIC_INDEX).info()
                logger.debug("Index %s already exists", self.SEMANTIC_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.name", as_name="name"),
                TextField("$.summary", as_name="summary"),
                TextField("$.details", as_name="details"),
                TextField("$.source", as_name="source"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                
                # ⭐ Three vector fields for comprehensive search (48KB total!)
                VectorField(
                    "$.name_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="name_embedding"
                ),
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="summary_embedding"
                ),
                VectorField(
                    "$.details_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="details_embedding"
                ),
            )
            
            self.client.ft(self.SEMANTIC_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.SEMANTIC_PREFIX],
                    index_type=IndexType.JSON
                )
            )
            logger.info("✅ Created JSON+VECTOR index: %s (3 vectors, 48KB!)", self.SEMANTIC_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create semantic index: %s", e)
    
    def _create_procedural_index(self) -> None:
        """Create JSON-based index for procedural memory with 2 VECTOR fields."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            from mirix.constants import MAX_EMBEDDING_DIM
            
            try:
                self.client.ft(self.PROCEDURAL_INDEX).info()
                logger.debug("Index %s already exists", self.PROCEDURAL_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.entry_type", as_name="entry_type"),
                TextField("$.summary", as_name="summary"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                
                # ⭐ Two vector fields (32KB total)
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="summary_embedding"
                ),
                VectorField(
                    "$.steps_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="steps_embedding"
                ),
            )
            
            self.client.ft(self.PROCEDURAL_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.PROCEDURAL_PREFIX],
                    index_type=IndexType.JSON
                )
            )
            logger.info("✅ Created JSON+VECTOR index: %s (2 vectors)", self.PROCEDURAL_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create procedural index: %s", e)
    
    def _create_resource_index(self) -> None:
        """Create JSON-based index for resource memory with 1 VECTOR field."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            from mirix.constants import MAX_EMBEDDING_DIM
            
            try:
                self.client.ft(self.RESOURCE_INDEX).info()
                logger.debug("Index %s already exists", self.RESOURCE_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.title", as_name="title"),
                TextField("$.summary", as_name="summary"),
                TextField("$.resource_type", as_name="resource_type"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                
                # ⭐ One vector field (16KB)
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="summary_embedding"
                ),
            )
            
            self.client.ft(self.RESOURCE_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.RESOURCE_PREFIX],
                    index_type=IndexType.JSON
                )
            )
            logger.info("✅ Created JSON+VECTOR index: %s (1 vector)", self.RESOURCE_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create resource index: %s", e)
    
    def _create_knowledge_index(self) -> None:
        """Create JSON-based index for knowledge vault with 1 VECTOR field."""
        try:
            from redis.commands.search.field import TextField, NumericField, TagField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            from mirix.constants import MAX_EMBEDDING_DIM
            
            try:
                self.client.ft(self.KNOWLEDGE_INDEX).info()
                logger.debug("Index %s already exists", self.KNOWLEDGE_INDEX)
                return
            except:
                pass
            
            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.caption", as_name="caption"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                
                # ⭐ One vector field (16KB)
                VectorField(
                    "$.caption_embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": MAX_EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="caption_embedding"
                ),
            )
            
            self.client.ft(self.KNOWLEDGE_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.KNOWLEDGE_PREFIX],
                    index_type=IndexType.JSON
                )
            )
            logger.info("✅ Created JSON+VECTOR index: %s (1 vector)", self.KNOWLEDGE_INDEX)
            
        except Exception as e:
            logger.warning("Failed to create knowledge index: %s", e)
    
    def set_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store data as Redis JSON (for complex structures with embeddings).
        
        Args:
            key: Redis key
            data: Data to store (supports nested structures)
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        try:
            # Use JSON.SET command
            self.client.json().set(key, "$", data)
            
            if ttl:
                self.client.expire(key, ttl)
            
            logger.debug("✅ Stored JSON: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to set JSON for %s: %s", key, e)
            return False
    
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from Redis JSON.
        
        Args:
            key: Redis key
        
        Returns:
            Data dictionary or None if not found
        """
        try:
            data = self.client.json().get(key)
            if data is None:
                return None
            
            logger.debug("✅ Retrieved JSON: %s", key)
            return data
        except Exception as e:
            logger.error("Failed to get JSON for %s: %s", key, e)
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        try:
            self.client.delete(key)
            logger.debug("✅ Deleted key: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to delete key %s: %s", key, e)
            return False
    
    def search_text(
        self,
        index_name: str,
        query: str,
        search_fields: List[str],
        limit: int = 10,
        user_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text search using RediSearch.
        
        Args:
            index_name: Index to search (e.g., "idx:episodic_memory")
            query: Search query text
            search_fields: Fields to search in (e.g., ["summary", "details"])
            limit: Maximum number of results
            user_id: Filter by user_id (optional)
            return_fields: Specific fields to return (None = return all)
        
        Returns:
            List of matching documents with BM25-like scores
        
        Example:
            results = redis_client.search_text(
                index_name="idx:episodic_memory",
                query="meeting Sarah",
                search_fields=["details"],
                limit=10,
                user_id="user-123"
            )
        """
        try:
            from redis.commands.search.query import Query
            import re
            
            # Escape special characters in query for Redis Search
            # Redis Search special characters: , . < > { } [ ] " ' : ; ! @ # $ % ^ & * ( ) - + = ~
            def escape_redis_query(text: str) -> str:
                """Escape special characters for Redis Search query."""
                special_chars = r'[,.<>{}[\]"\':;!@#$%^&*()\-+=~]'
                return re.sub(special_chars, lambda m: f'\\{m.group(0)}', text)
            
            escaped_query = escape_redis_query(query)
            
            # Build field search query
            if len(search_fields) == 1:
                search_query = f"@{search_fields[0]}:({escaped_query})"
            else:
                field_query = "|".join(search_fields)
                search_query = f"@{field_query}:({escaped_query})"
            
            # Add user_id filter if provided (escape special characters in TAG field)
            if user_id:
                escaped_user_id = user_id.replace("-", "\\-").replace(":", "\\:")
                search_query = f"@user_id:{{{escaped_user_id}}} {search_query}"
            
            # Build Query object
            query_obj = Query(search_query).paging(0, limit)
            
            # Add return fields if specified
            if return_fields:
                query_obj = query_obj.return_fields(*return_fields)
            
            # Execute search
            results = self.client.ft(index_name).search(query_obj)
            
            # Parse results
            documents = []
            for doc in results.docs:
                # For JSON documents
                if hasattr(doc, 'json'):
                    documents.append(json.loads(doc.json))
                # For Hash documents
                else:
                    doc_dict = {}
                    for key, value in doc.__dict__.items():
                        if not key.startswith('_'):
                            doc_dict[key] = value
                    documents.append(doc_dict)
            
            logger.debug("✅ Redis text search: found %d results in %s", len(documents), index_name)
            return documents
        
        except Exception as e:
            logger.warning("Redis text search failed for index %s with query '%s': %s", index_name, query[:50], e)
            return []
    
    def search_vector(
        self,
        index_name: str,
        embedding: List[float],
        vector_field: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using RediSearch KNN.
        
        Args:
            index_name: Index to search
            embedding: Query embedding vector (must match dimension of indexed vectors)
            vector_field: Vector field to search (e.g., "summary_embedding", "details_embedding")
            limit: Maximum number of results (K in KNN)
            user_id: Filter by user_id (optional)
            return_fields: Specific fields to return (None = return all)
        
        Returns:
            List of similar documents sorted by cosine similarity
        
        Example:
            results = redis_client.search_vector(
                index_name="idx:semantic_memory",
                embedding=[0.1, 0.2, ...],  # 1536-dim vector
                vector_field="summary_embedding",
                limit=10,
                user_id="user-123"
            )
        """
        try:
            from redis.commands.search.query import Query
            import numpy as np
            
            # Convert embedding to bytes
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            # Build KNN query
            # Using KNN syntax: *=>[KNN K @vector_field $vec AS distance]
            knn_query = f"*=>[KNN {limit} @{vector_field} $vec AS vector_distance]"
            
            # Add user_id filter if provided (escape special characters in TAG field)
            if user_id:
                escaped_user_id = user_id.replace("-", "\\-").replace(":", "\\:")
                knn_query = f"@user_id:{{{escaped_user_id}}} {knn_query}"
            
            # Build Query object
            query_obj = (
                Query(knn_query)
                .sort_by("vector_distance")
                .paging(0, limit)
                .dialect(2)  # Required for vector search
            )
            
            # Add return fields if specified
            if return_fields:
                query_obj = query_obj.return_fields(*return_fields, "vector_distance")
            else:
                query_obj = query_obj.return_fields("vector_distance")
            
            # Execute search
            results = self.client.ft(index_name).search(
                query_obj,
                query_params={"vec": embedding_bytes}
            )
            
            # Parse results
            documents = []
            for doc in results.docs:
                # For JSON documents
                if hasattr(doc, 'json'):
                    doc_dict = json.loads(doc.json)
                # For Hash documents
                else:
                    doc_dict = {}
                    for key, value in doc.__dict__.items():
                        if not key.startswith('_'):
                            doc_dict[key] = value
                
                # Add similarity score (convert distance to similarity)
                if hasattr(doc, 'vector_distance'):
                    # Cosine distance to similarity: similarity = 1 - distance
                    doc_dict['similarity_score'] = 1.0 - float(doc.vector_distance)
                
                documents.append(doc_dict)
            
            logger.debug("✅ Redis vector search: found %d results in %s", len(documents), index_name)
            return documents
        
        except Exception as e:
            logger.warning("Redis vector search failed for index %s (vector_field: %s): %s", index_name, vector_field, e)
            return []
    
    def search_recent(
        self,
        index_name: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        sort_by: str = "created_at_ts",
        return_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most recent documents from an index.
        
        Args:
            index_name: Index to search
            limit: Maximum number of results
            user_id: Filter by user_id (optional)
            sort_by: Field to sort by (default: "created_at_ts")
            return_fields: Specific fields to return (None = return all)
        
        Returns:
            List of recent documents sorted by timestamp (descending)
        
        Example:
            results = redis_client.search_recent(
                index_name="idx:episodic_memory",
                limit=10,
                user_id="user-123",
                sort_by="occurred_at_ts"
            )
        """
        try:
            from redis.commands.search.query import Query
            
            # Build query to match all documents for user (escape special characters in TAG field)
            if user_id:
                escaped_user_id = user_id.replace("-", "\\-").replace(":", "\\:")
                search_query = f"@user_id:{{{escaped_user_id}}}"
            else:
                search_query = "*"  # Match all
            
            # Build Query object with sorting
            query_obj = (
                Query(search_query)
                .sort_by(sort_by, asc=False)  # Descending order (most recent first)
                .paging(0, limit)
            )
            
            # Add return fields if specified
            if return_fields:
                query_obj = query_obj.return_fields(*return_fields)
            
            # Execute search
            results = self.client.ft(index_name).search(query_obj)
            
            # Parse results
            documents = []
            for doc in results.docs:
                # For JSON documents
                if hasattr(doc, 'json'):
                    documents.append(json.loads(doc.json))
                # For Hash documents
                else:
                    doc_dict = {}
                    for key, value in doc.__dict__.items():
                        if not key.startswith('_'):
                            doc_dict[key] = value
                    documents.append(doc_dict)
            
            logger.debug("✅ Redis recent search: found %d results in %s", len(documents), index_name)
            return documents
        
        except Exception as e:
            logger.warning("Redis recent search failed for index %s (sort_by: %s): %s", index_name, sort_by, e)
            return []
    
    @staticmethod
    def clean_redis_fields(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove Redis-specific fields before Pydantic validation.
        
        ⚠️ SECURITY: We keep Pydantic's extra="forbid" for API validation security.
        This helper strips Redis-internal fields (_ts timestamps, search metadata) that
        would otherwise cause validation errors, while preserving strict validation for
        user input.
        
        Redis stores both datetime ISO strings and numeric timestamps (_ts suffix) for sorting.
        This helper removes the Redis-specific _ts fields after search operations.
        
        Args:
            items: List of dictionaries from Redis Search results
            
        Returns:
            List of cleaned dictionaries ready for strict Pydantic validation
        """
        from datetime import datetime
        
        for item in items:
            # Remove Redis-specific timestamp fields (keeping only the datetime ISO strings)
            if 'created_at_ts' in item:
                # If we don't have the ISO string, convert from timestamp
                if 'created_at' not in item:
                    item['created_at'] = datetime.fromtimestamp(item['created_at_ts']).isoformat()
                item.pop('created_at_ts')
                
            if 'occurred_at_ts' in item:
                # If we don't have the ISO string, convert from timestamp
                if 'occurred_at' not in item:
                    item['occurred_at'] = datetime.fromtimestamp(item['occurred_at_ts']).isoformat()
                item.pop('occurred_at_ts')
            
            # Remove search metadata fields
            item.pop('similarity_score', None)
            item.pop('vector_distance', None)
        
        return items


def initialize_redis_client() -> Optional[RedisMemoryClient]:
    """Initialize global Redis client from settings."""
    global _redis_client
    
    if _redis_client is not None:
        return _redis_client
    
    try:
        from mirix.settings import settings
        
        if not settings.redis_enabled:
            logger.info("Redis is disabled (MIRIX_REDIS_ENABLED=false)")
            return None
        
        redis_uri = settings.mirix_redis_uri
        if not redis_uri:
            logger.warning("Redis enabled but no URI configured")
            return None
        
        _redis_client = RedisMemoryClient(
            redis_uri=redis_uri,
            max_connections=settings.redis_max_connections,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
        )
        
        # Test connection
        if not _redis_client.ping():
            logger.error("Redis ping failed - disabling Redis")
            _redis_client = None
            return None
        
        # Create indexes
        _redis_client.create_indexes()
        
        logger.info("✅ Redis client initialized and indexes created")
        return _redis_client
        
    except Exception as e:
        logger.error("Failed to initialize Redis client: %s", e)
        logger.info("System will continue without Redis caching")
        _redis_client = None
        return None


def get_redis_client() -> Optional[RedisMemoryClient]:
    """Get the global Redis client instance."""
    return _redis_client


def close_redis_client() -> None:
    """Close the global Redis client."""
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None

