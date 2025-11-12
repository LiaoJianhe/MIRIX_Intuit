"""
ORM model for external domain agents that register to use MIRIX
This is separate from the internal 'agents' table used by MIRIX's memory agents
"""
from typing import TYPE_CHECKING, List, Optional
from datetime import datetime

from sqlalchemy import JSON, String, Boolean, DateTime, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.registered_agent import RegisteredAgent as PydanticRegisteredAgent

if TYPE_CHECKING:
    from mirix.orm.organization import Organization


class RegisteredAgent(SqlalchemyBase, OrganizationMixin):
    """
    External domain agents that register to use MIRIX as a memory backend.
    
    These are different from MIRIX's internal agents (chat_agent, episodic_memory_agent, etc.)
    which are stored in the 'agents' table and handle memory operations.
    
    RegisteredAgent represents:
    - Your custom domain agents (customer service, sales, support, etc.)
    - API clients that consume MIRIX services
    - Applications that need permission-based access to MIRIX
    
    Attributes:
        id: Unique identifier for the registered agent
        name: Agent name (e.g., "customer_service_agent")
        description: What the agent does
        scope: Domain/category (e.g., "tax", "bookkeeping", "sales", "marketing")
        api_key: Authentication key for API calls
        permissions: List of permissions (e.g., ["read", "write", "delete"])
        active: Whether the agent is currently active
        rate_limit: API rate limit (requests per minute)
        metadata_: Additional configuration (JSON)
        last_activity: Last time agent made an API call
    """

    __tablename__ = "registered_agents"
    __pydantic_model__ = PydanticRegisteredAgent
    
    # Create indexes for common queries
    __table_args__ = (
        Index("ix_registered_agents_api_key", "api_key"),
        Index("ix_registered_agents_scope", "scope"),
        Index("ix_registered_agents_active", "active"),
    )

    # Basic Information
    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="The name of the registered agent"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Description of what this agent does"
    )
    
    scope: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Domain/category: tax, bookkeeping, sales, marketing, support, etc."
    )
    
    # Authentication & Authorization
    api_key: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True,
        doc="API key for authentication"
    )
    
    permissions: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=["read"],
        doc="List of permissions: read, write, delete, admin, etc."
    )
    
    # Status & Activity
    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether the agent is currently active"
    )
    
    rate_limit: Mapped[int] = mapped_column(
        nullable=False,
        default=100,
        doc="Rate limit in requests per minute"
    )
    
    last_activity: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        doc="Last time the agent made an API call"
    )
    
    # Additional Configuration
    metadata_: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        doc="Additional metadata and configuration"
    )
    
    # Allowed scopes for data access (optional restriction)
    allowed_scopes: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Optional: Restrict agent to specific data scopes"
    )
    
    # Relationship
    organization: Mapped["Organization"] = relationship(
        "Organization",
        back_populates="registered_agents"
    )
    
    def has_permission(self, permission: str) -> bool:
        """Check if agent has a specific permission"""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if agent has any of the specified permissions"""
        return any(p in self.permissions for p in permissions)
    
    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if agent has all of the specified permissions"""
        return all(p in self.permissions for p in permissions)
    
    def is_scope_allowed(self, scope: str) -> bool:
        """Check if agent can access a specific scope"""
        if self.allowed_scopes is None:
            return True  # No restrictions
        return scope in self.allowed_scopes
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()



