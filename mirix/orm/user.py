from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.user import User as PydanticUser

if TYPE_CHECKING:
    from mirix.orm import Organization


class User(SqlalchemyBase):
    """User ORM class — IMPORTANT: users are GLOBAL, not per-org. Read this.

    Identity is the ``id`` column alone, NOT a composite (organization_id, id).
    This matches IPS-Relational's id-only PK invariant — the platform does not
    support composite primary keys, so any design relying on
    "same id under different orgs" cannot work there. PostgreSQL is kept in
    lockstep (id-only PK here too) so both backends behave identically.

    What this means in practice
    ---------------------------
    * The ``users`` row is a SHARED STUB across all orgs.
      A call to ``save(user_id="1234", organization_id="org-A")`` creates one
      row with ``id="1234"``. A subsequent ``save(user_id="1234",
      organization_id="org-B")`` does NOT create a second row — the existing
      one is reused (see ``UserManager.create_user``'s catch-and-swallow).
    * The ``organization_id`` column on this row records who happened to
      create it. It is NOT a tenancy key — never filter on it for org
      isolation.
    * Per-org isolation lives on CHILD tables (``episodic_memory``,
      ``block``, ``message``, ``memory_source``, etc.). Every child row
      carries its own ``organization_id``. A search by
      ``(user_id="1234", organization_id="org-A")`` finds only org-A's
      child rows, even though the parent ``users`` row is shared.
    * Two callers in different orgs can use the same user id (``"1234"``,
      ``"alice"``, the well-known ``ADMIN_USER_ID``) and stay correctly
      isolated. They share an identity row; they do not share data.

    What NOT to do
    --------------
    * Do NOT add per-org user-level data here (name override, preferences,
      scope flags, etc.). The row is shared. If you need per-(user, org)
      state, put it on a child table with its own ``organization_id``.
    * Do NOT delete users in cleanup/teardown without confirming no other
      org references them — child rows from other orgs still FK onto this
      ``users.id``. See ``cleanup_ips_test_data.py``: users are
      intentionally preserved as namespace stubs.
    * Do NOT add ``AND organization_id = :organizationId`` to NQs that
      look up by user id. ``user_manager.get_user_by_id`` is id-only by
      design.

    History
    -------
    Earlier VEPAGE-1155 work made this a composite PK ``(organization_id, id)``
    to allow the same well-known id (``ADMIN_USER_ID``) under multiple orgs.
    IPS-R cannot enforce composite PKs, so that design collided on
    ``users_pkey`` in production. Option 1 (this design) reverts to
    id-only with shared rows, matching the platform.
    """

    __tablename__ = "users"
    __pydantic_model__ = PydanticUser

    organization_id: Mapped[str] = mapped_column(
        String, ForeignKey("organizations.id"), nullable=False
    )

    name: Mapped[str] = mapped_column(nullable=False, doc="The display name of the user.")
    status: Mapped[str] = mapped_column(nullable=False, doc="Whether the user is active or not.")
    timezone: Mapped[str] = mapped_column(nullable=False, doc="The timezone of the user.")
    is_admin: Mapped[bool] = mapped_column(nullable=False, default=False, doc="Whether this is an admin user.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="users")

    # TODO: Add this back later potentially
    # tokens: Mapped[List["Token"]] = relationship("Token", back_populates="user", doc="the tokens associated with this user.")
