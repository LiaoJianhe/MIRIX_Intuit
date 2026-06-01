from typing import Optional
from uuid import UUID

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.base import Base


def is_valid_uuid4(uuid_string: str) -> bool:
    """Check if a string is a valid UUID4."""
    try:
        uuid_obj = UUID(uuid_string)
        return uuid_obj.version == 4
    except ValueError:
        return False


class OrganizationMixin(Base):
    """Mixin for models that belong to an organization."""

    __abstract__ = True

    organization_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("organizations.id"), nullable=True)


class UserMixin(Base):
    """Mixin for models that belong to a user.

    The FK to users is composite (organization_id, user_id) -> (users.organization_id,
    users.id) and cannot be declared here because it needs the consumer's
    organization_id column. Each consumer adds a ForeignKeyConstraint in __table_args__.
    """

    __abstract__ = True

    user_id: Mapped[str] = mapped_column(String)


class AgentMixin(Base):
    """Mixin for models that belong to an agent."""

    __abstract__ = True

    agent_id: Mapped[str] = mapped_column(String, ForeignKey("agents.id", ondelete="CASCADE"))
