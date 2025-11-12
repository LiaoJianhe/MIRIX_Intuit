"""
Pydantic schemas for registered external agents
"""
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class RegisteredAgentBase(BaseModel):
    """Base schema for registered agents"""
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    scope: str = Field(..., description="Domain/category (tax, sales, etc.)")
    permissions: List[str] = Field(
        default=["read"],
        description="List of permissions (read, write, delete, etc.)"
    )
    active: bool = Field(default=True, description="Whether agent is active")
    rate_limit: int = Field(default=100, description="API rate limit (req/min)")
    metadata_: Optional[dict] = Field(None, description="Additional configuration")
    allowed_scopes: Optional[List[str]] = Field(
        None,
        description="Optional: Restrict to specific data scopes"
    )


class RegisterAgentRequest(RegisteredAgentBase):
    """Request schema for registering a new agent"""
    pass


class RegisterAgentResponse(BaseModel):
    """Response schema for agent registration"""
    success: bool
    message: str
    agent: Optional["RegisteredAgent"] = None
    api_key: Optional[str] = None  # Only returned on creation


class RegisteredAgent(RegisteredAgentBase):
    """Full registered agent schema"""
    id: str
    api_key: str
    organization_id: str
    last_activity: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    is_deleted: bool = False
    
    class Config:
        from_attributes = True


class UpdateRegisteredAgentRequest(BaseModel):
    """Request schema for updating a registered agent"""
    name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[str]] = None
    active: Optional[bool] = None
    rate_limit: Optional[int] = None
    metadata_: Optional[dict] = None
    allowed_scopes: Optional[List[str]] = None


class ValidateAgentRequest(BaseModel):
    """Request schema for validating agent permissions"""
    api_key: str
    required_permissions: List[str]
    scope: Optional[str] = None


class ValidateAgentResponse(BaseModel):
    """Response schema for permission validation"""
    valid: bool
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    message: str
    missing_permissions: Optional[List[str]] = None



