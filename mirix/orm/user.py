from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.user import User as PydanticUser

if TYPE_CHECKING:
    from mirix.orm import Organization


class User(SqlalchemyBase):
    """User ORM class - users are organization-scoped.

    Identity is the composite (organization_id, id): the same id may recur in
    different organizations (e.g. the well-known admin user id), so id alone is
    not unique. organization_id is part of the primary key rather than a plain
    OrganizationMixin column.
    """

    __tablename__ = "users"
    __pydantic_model__ = PydanticUser

    organization_id: Mapped[str] = mapped_column(
        String, ForeignKey("organizations.id"), primary_key=True
    )

    name: Mapped[str] = mapped_column(nullable=False, doc="The display name of the user.")
    status: Mapped[str] = mapped_column(nullable=False, doc="Whether the user is active or not.")
    timezone: Mapped[str] = mapped_column(nullable=False, doc="The timezone of the user.")
    is_admin: Mapped[bool] = mapped_column(nullable=False, default=False, doc="Whether this is an admin user.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")

    # TODO: Add this back later potentially
    # tokens: Mapped[List["Token"]] = relationship("Token", back_populates="user", doc="the tokens associated with this user.")
