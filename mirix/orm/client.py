from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.client import Client as PydanticClient

if TYPE_CHECKING:
    from mirix.orm import Organization
    from mirix.orm.client_api_key import ClientApiKey


class Client(SqlalchemyBase):
    """Client ORM class - represents a client application.

    Identity is the composite (organization_id, id): the same id may recur in
    different organizations (e.g. the well-known default client id), so id alone
    is not unique. organization_id is part of the primary key rather than a plain
    OrganizationMixin column.
    """

    __tablename__ = "clients"
    __pydantic_model__ = PydanticClient

    organization_id: Mapped[str] = mapped_column(
        String, ForeignKey("organizations.id"), primary_key=True
    )

    # Basic fields
    name: Mapped[str] = mapped_column(nullable=False, doc="The display name of the client application.")
    status: Mapped[str] = mapped_column(nullable=False, doc="Whether the client is active or not.")
    write_scope: Mapped[Optional[str]] = mapped_column(
        nullable=True, default=None, doc="Scope for writing memories (null = read-only)."
    )
    read_scopes: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list, doc="Scopes for reading memories."
    )

    # Message retention
    message_set_retention_count: Mapped[Optional[int]] = mapped_column(
        nullable=True, default=0, doc="Number of input message-sets to retain per (agent, user). 0 = no retention."
    )

    # Dashboard authentication fields
    email: Mapped[Optional[str]] = mapped_column(
        nullable=True, unique=True, index=True, doc="Email address for dashboard login."
    )
    password_hash: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="Hashed password for dashboard login (bcrypt)."
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(nullable=True, doc="Last dashboard login time.")

    # Relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="clients")
    api_keys: Mapped[List["ClientApiKey"]] = relationship(
        "ClientApiKey",
        back_populates="client",
        cascade="all, delete-orphan",
        lazy="selectin",
        foreign_keys="[ClientApiKey.organization_id, ClientApiKey.client_id]",
        primaryjoin=(
            "and_(Client.organization_id == foreign(ClientApiKey.organization_id), "
            "Client.id == foreign(ClientApiKey.client_id))"
        ),
    )
