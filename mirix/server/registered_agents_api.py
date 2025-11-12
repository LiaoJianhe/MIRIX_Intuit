"""
API endpoints for external agent registration and permission management
"""
import secrets
from typing import List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Header
from sqlalchemy.orm import Session

from mirix.orm.registered_agent import RegisteredAgent
from mirix.schemas.registered_agent import (
    RegisterAgentRequest,
    RegisterAgentResponse,
    RegisteredAgent as PydanticRegisteredAgent,
    UpdateRegisteredAgentRequest,
    ValidateAgentRequest,
    ValidateAgentResponse
)

router = APIRouter(prefix="/registered-agents", tags=["Agent Registration"])

# This will be injected from the main server
# For now, using a placeholder
def get_db():
    """Dependency to get database session"""
    from mirix.server.server import db_context
    with db_context() as db:
        yield db


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"ma_{secrets.token_urlsafe(32)}"  # ma_ = mirix agent


@router.post("/register", response_model=RegisterAgentResponse)
async def register_agent(
    request: RegisterAgentRequest,
    db: Session = Depends(get_db)
):
    """
    Register a new external domain agent
    
    This creates an API key and stores agent details with permissions.
    
    Example:
        POST /registered-agents/register
        {
            "name": "customer_service_agent",
            "description": "Handles customer support inquiries",
            "scope": "support",
            "permissions": ["read", "write"],
            "rate_limit": 100
        }
    """
    try:
        # Check if agent name already exists
        existing = db.query(RegisteredAgent).filter(
            RegisteredAgent.name == request.name
        ).first()
        
        if existing:
            return RegisterAgentResponse(
                success=False,
                message=f"Agent with name '{request.name}' already exists",
                agent=None,
                api_key=None
            )
        
        # Generate API key
        api_key = generate_api_key()
        
        # Create new registered agent
        agent = RegisteredAgent(
            name=request.name,
            description=request.description,
            scope=request.scope,
            api_key=api_key,
            permissions=request.permissions,
            active=request.active,
            rate_limit=request.rate_limit,
            metadata_=request.metadata_,
            allowed_scopes=request.allowed_scopes,
            # organization_id will be set by OrganizationMixin
        )
        
        db.add(agent)
        db.commit()
        db.refresh(agent)
        
        return RegisterAgentResponse(
            success=True,
            message=f"Agent '{request.name}' registered successfully",
            agent=PydanticRegisteredAgent.model_validate(agent),
            api_key=api_key  # Only returned once!
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error registering agent: {str(e)}")


@router.post("/validate", response_model=ValidateAgentResponse)
async def validate_agent_permissions(
    request: ValidateAgentRequest,
    db: Session = Depends(get_db)
):
    """
    Validate agent has required permissions
    
    This should be called before processing any agent request.
    
    Example:
        POST /registered-agents/validate
        {
            "api_key": "ma_abc123...",
            "required_permissions": ["write"],
            "scope": "support"
        }
    """
    try:
        # Find agent by API key
        agent = db.query(RegisteredAgent).filter(
            RegisteredAgent.api_key == request.api_key
        ).first()
        
        if not agent:
            return ValidateAgentResponse(
                valid=False,
                message="Invalid API key",
                missing_permissions=None
            )
        
        # Check if agent is active
        if not agent.active:
            return ValidateAgentResponse(
                valid=False,
                agent_id=agent.id,
                agent_name=agent.name,
                message="Agent is inactive",
                missing_permissions=None
            )
        
        # Check permissions
        missing_permissions = [
            perm for perm in request.required_permissions
            if perm not in agent.permissions
        ]
        
        if missing_permissions:
            return ValidateAgentResponse(
                valid=False,
                agent_id=agent.id,
                agent_name=agent.name,
                message=f"Missing required permissions: {', '.join(missing_permissions)}",
                missing_permissions=missing_permissions
            )
        
        # Check scope if provided
        if request.scope and not agent.is_scope_allowed(request.scope):
            return ValidateAgentResponse(
                valid=False,
                agent_id=agent.id,
                agent_name=agent.name,
                message=f"Agent not allowed to access scope: {request.scope}",
                missing_permissions=None
            )
        
        # Update last activity
        agent.update_activity()
        db.commit()
        
        return ValidateAgentResponse(
            valid=True,
            agent_id=agent.id,
            agent_name=agent.name,
            message="Agent authorized",
            missing_permissions=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating agent: {str(e)}")


@router.get("/list", response_model=List[PydanticRegisteredAgent])
async def list_registered_agents(
    scope: str = None,
    active: bool = None,
    db: Session = Depends(get_db)
):
    """
    List all registered agents
    
    Query parameters:
        - scope: Filter by scope (e.g., "tax", "sales")
        - active: Filter by active status (true/false)
    """
    try:
        query = db.query(RegisteredAgent)
        
        if scope:
            query = query.filter(RegisteredAgent.scope == scope)
        
        if active is not None:
            query = query.filter(RegisteredAgent.active == active)
        
        agents = query.all()
        
        return [PydanticRegisteredAgent.model_validate(agent) for agent in agents]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")


@router.get("/{agent_id}", response_model=PydanticRegisteredAgent)
async def get_registered_agent(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """Get details of a specific registered agent"""
    try:
        agent = db.query(RegisteredAgent).filter(
            RegisteredAgent.id == agent_id
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
        return PydanticRegisteredAgent.model_validate(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent: {str(e)}")


@router.patch("/{agent_id}", response_model=PydanticRegisteredAgent)
async def update_registered_agent(
    agent_id: str,
    request: UpdateRegisteredAgentRequest,
    db: Session = Depends(get_db)
):
    """
    Update a registered agent's details
    
    Can update: name, description, permissions, active status, rate limit, etc.
    """
    try:
        agent = db.query(RegisteredAgent).filter(
            RegisteredAgent.id == agent_id
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
        # Update fields
        if request.name is not None:
            agent.name = request.name
        if request.description is not None:
            agent.description = request.description
        if request.permissions is not None:
            agent.permissions = request.permissions
        if request.active is not None:
            agent.active = request.active
        if request.rate_limit is not None:
            agent.rate_limit = request.rate_limit
        if request.metadata_ is not None:
            agent.metadata_ = request.metadata_
        if request.allowed_scopes is not None:
            agent.allowed_scopes = request.allowed_scopes
        
        db.commit()
        db.refresh(agent)
        
        return PydanticRegisteredAgent.model_validate(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")


@router.delete("/{agent_id}")
async def delete_registered_agent(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """Delete (deactivate) a registered agent"""
    try:
        agent = db.query(RegisteredAgent).filter(
            RegisteredAgent.id == agent_id
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
        # Soft delete - just mark as inactive
        agent.active = False
        agent.is_deleted = True
        db.commit()
        
        return {
            "success": True,
            "message": f"Agent '{agent.name}' has been deactivated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")


# Middleware/Dependency for protected endpoints
async def verify_agent_permission(
    x_api_key: str = Header(..., description="Agent API key"),
    required_permission: str = "read",
    db: Session = Depends(get_db)
) -> RegisteredAgent:
    """
    Dependency to verify agent has required permission
    
    Usage in protected endpoints:
        @router.post("/some-protected-endpoint")
        async def protected_endpoint(
            agent: RegisteredAgent = Depends(verify_agent_permission)
        ):
            # agent is validated and has required permission
            ...
    """
    agent = db.query(RegisteredAgent).filter(
        RegisteredAgent.api_key == x_api_key
    ).first()
    
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not agent.active:
        raise HTTPException(status_code=403, detail="Agent is inactive")
    
    if not agent.has_permission(required_permission):
        raise HTTPException(
            status_code=403,
            detail=f"Agent lacks required permission: {required_permission}"
        )
    
    # Update last activity
    agent.update_activity()
    db.commit()
    
    return agent



