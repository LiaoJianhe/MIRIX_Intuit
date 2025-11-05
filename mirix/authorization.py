"""
Authorization Module for Domain Agent Access Control

This module provides:
1. Permission checking for API endpoints
2. Scope extraction for business rules
3. Context management for scoped operations
"""
from typing import Optional, List, Callable
from functools import wraps
from fastapi import HTTPException, Header, Request
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Authorization Context - Stores agent info for request lifecycle
# ============================================================================

class AuthContext:
    """Holds authorization context for the current request"""
    
    def __init__(self, agent_id: str, scope: str, permissions: List[str]):
        self.agent_id = agent_id
        self.scope = scope
        self.permissions = permissions
    
    def has_permission(self, permission: str) -> bool:
        """Check if agent has specific permission"""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if agent has any of the specified permissions"""
        return any(p in self.permissions for p in permissions)


# ============================================================================
# Agent Validator - Fetches and validates agent from database
# ============================================================================

class AgentValidator:
    """Validates registered agents and their permissions"""
    
    @staticmethod
    def validate_agent(db, agent_id: str, required_permissions: List[str]) -> AuthContext:
        """
        Validate agent and check permissions
        
        Args:
            db: Database session
            agent_id: Agent identifier
            required_permissions: List of required permissions
        
        Returns:
            AuthContext with agent details
        
        Raises:
            HTTPException: If validation fails
        """
        # Fetch agent from database
        result = db.execute(
            text("""
                SELECT agent_id, scope, permissions, active
                FROM app.registered_agents
                WHERE agent_id = :agent_id
            """),
            {"agent_id": agent_id}
        )
        
        row = result.fetchone()
        
        if not row:
            raise HTTPException(
                status_code=401,
                detail=f"Agent '{agent_id}' not found. Please register the agent first."
            )
        
        agent_id_db, scope, permissions, active = row
        
        # Check if agent is active
        if not active:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_id}' is inactive. Please contact administrator."
            )
        
        # Check permissions
        missing_permissions = [p for p in required_permissions if p not in permissions]
        
        if missing_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Agent '{agent_id}' lacks required permissions: {', '.join(missing_permissions)}. "
                       f"Current permissions: {', '.join(permissions)}"
            )
        
        logger.info(f"Agent '{agent_id}' authorized with scope '{scope}'")
        
        return AuthContext(
            agent_id=agent_id_db,
            scope=scope,
            permissions=permissions
        )


# ============================================================================
# Authorization Decorator - For protecting API endpoints
# ============================================================================

def require_permissions(required_permissions: List[str]):
    """
    Decorator to require specific permissions for an endpoint
    
    Usage:
        @app.post("/some-endpoint")
        @require_permissions(["read", "write"])
        async def some_endpoint(
            request: Request,
            agent_id: str = Header(..., alias="X-Agent-ID")
        ):
            # Access auth context
            auth = request.state.auth
            scope = auth.scope
            # ... your code
    
    Args:
        required_permissions: List of required permissions
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and agent_id from kwargs
            request = kwargs.get("request")
            agent_id = kwargs.get("agent_id")
            
            if not request:
                raise HTTPException(
                    status_code=500,
                    detail="Request object not found in endpoint parameters"
                )
            
            if not agent_id:
                raise HTTPException(
                    status_code=401,
                    detail="X-Agent-ID header is required"
                )
            
            # Get database session
            from mirix.server.server import db_context
            
            with db_context() as db:
                # Validate agent and permissions
                auth_context = AgentValidator.validate_agent(
                    db, agent_id, required_permissions
                )
                
                # Store in request state for access in endpoint
                request.state.auth = auth_context
                
                # Call the actual endpoint
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# Scope Context Manager - For scoped MIRIX operations
# ============================================================================

class ScopeContext:
    """
    Manages scope-based operations in MIRIX
    
    This ensures that memories are tagged with the agent's scope
    for proper data isolation and business rules.
    """
    
    def __init__(self, scope: str):
        self.scope = scope
    
    def add_scope_metadata(self, metadata: Optional[dict] = None) -> dict:
        """
        Add scope to metadata for MIRIX operations
        
        Usage:
            scope_ctx = ScopeContext(auth.scope)
            mirix.add(
                message,
                user_id=user_id,
                metadata=scope_ctx.add_scope_metadata({"key": "value"})
            )
        """
        if metadata is None:
            metadata = {}
        
        metadata["agent_scope"] = self.scope
        return metadata
    
    def should_access_memory(self, memory_scope: Optional[str]) -> bool:
        """
        Business rule: Can this agent access a memory with given scope?
        
        Default rule: Agent can only access memories from same scope
        Override this method for custom business rules
        """
        if memory_scope is None:
            return True  # No scope restriction
        
        return memory_scope == self.scope
    
    def filter_memories_by_scope(self, memories: List[dict]) -> List[dict]:
        """Filter memories based on scope access rules"""
        return [
            mem for mem in memories
            if self.should_access_memory(mem.get("metadata", {}).get("agent_scope"))
        ]


# ============================================================================
# Helper Functions
# ============================================================================

async def get_auth_context(request: Request) -> AuthContext:
    """
    Get authorization context from request
    
    Usage in endpoints:
        auth = await get_auth_context(request)
        scope = auth.scope
        agent_id = auth.agent_id
    """
    if not hasattr(request.state, "auth"):
        raise HTTPException(
            status_code=500,
            detail="Authorization context not found. Did you forget @require_permissions?"
        )
    
    return request.state.auth


def check_scope_access(agent_scope: str, memory_scope: Optional[str]) -> bool:
    """
    Business rule for scope-based access control
    
    Current rule: Agent can only access memories from same scope
    
    Customize this function for your business rules:
    - Cross-scope access for admin agents
    - Hierarchical scopes (e.g., SALES can access MARKETING)
    - etc.
    """
    if memory_scope is None:
        return True
    
    return agent_scope == memory_scope


# ============================================================================
# Example Usage
# ============================================================================

"""
Example: Using decorator in FastAPI endpoint

@app.post("/api/chat")
@require_permissions(["read", "write"])
async def chat_endpoint(
    request: Request,
    message: str,
    customer_id: str,
    agent_id: str = Header(..., alias="X-Agent-ID")
):
    # Get authorization context
    auth = await get_auth_context(request)
    
    # Use scope for business logic
    scope_ctx = ScopeContext(auth.scope)
    
    # ... your chat logic ...
    
    # Store memory with scope metadata
    mirix.add(
        message,
        user_id=mirix_user_id,
        metadata=scope_ctx.add_scope_metadata()
    )
    
    return {"response": response}
"""



