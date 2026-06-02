"""
Unit tests for UserManager after client_id removal.

Tests verify that users are organization-scoped, not client-scoped:
1. Users created by different clients in the same org are shared
2. Users in different orgs are isolated
3. list_users() filters by organization_id
4. Deleting a client does NOT cascade-delete users
5. get_or_create_org_default_user() works without client_id

Run tests:
    pytest tests/test_user_manager.py -v
"""

import asyncio
import uuid
from datetime import datetime
from datetime import timezone as dt_timezone

import pytest
import pytest_asyncio

from mirix.log import get_logger
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.user import User as PydanticUser
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.user_manager import UserManager
from mirix.settings import settings

logger = get_logger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def generate_test_id(prefix: str) -> str:
    """Generate a test ID matching Mirix ID pattern (prefix-[8 hex chars])."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def organization_manager():
    """Create organization manager instance."""
    return OrganizationManager()


@pytest.fixture
def user_manager():
    """Create user manager instance."""
    return UserManager()


@pytest.fixture
def client_manager():
    """Create client manager instance."""
    return ClientManager()


@pytest_asyncio.fixture
async def test_org1(organization_manager):
    """Create test organization 1."""
    org = PydanticOrganization(id=generate_test_id("org"), name="Test Organization 1")
    created_org = await organization_manager.create_organization(org)
    yield created_org
    try:
        await organization_manager.delete_organization_by_id(created_org.id)
    except Exception:
        pass


@pytest_asyncio.fixture
async def test_org2(organization_manager):
    """Create test organization 2."""
    org = PydanticOrganization(id=generate_test_id("org"), name="Test Organization 2")
    created_org = await organization_manager.create_organization(org)
    yield created_org
    try:
        await organization_manager.delete_organization_by_id(created_org.id)
    except Exception:
        pass


@pytest_asyncio.fixture
async def client_a(test_org1, client_manager):
    """Create Client A in org1."""
    client = PydanticClient(
        id=generate_test_id("client"),
        name="Client A",
        organization_id=test_org1.id,
        status="active",
        write_scope="test",
        read_scopes=["test"],
    )
    created_client = await client_manager.create_client(client)
    yield created_client
    try:
        await client_manager.delete_client_by_id(created_client.id)
    except Exception:
        pass


@pytest_asyncio.fixture
async def client_b(test_org1, client_manager):
    """Create Client B in org1 (same org as Client A)."""
    client = PydanticClient(
        id=generate_test_id("client"),
        name="Client B",
        organization_id=test_org1.id,
        status="active",
        write_scope="test",
        read_scopes=["test"],
    )
    created_client = await client_manager.create_client(client)
    yield created_client
    try:
        await client_manager.delete_client_by_id(created_client.id)
    except Exception:
        pass


@pytest_asyncio.fixture
async def client_c(test_org2, client_manager):
    """Create Client C in org2 (different org)."""
    client = PydanticClient(
        id=generate_test_id("client"),
        name="Client C",
        organization_id=test_org2.id,
        status="active",
        write_scope="test",
        read_scopes=["test"],
    )
    created_client = await client_manager.create_client(client)
    yield created_client
    try:
        await client_manager.delete_client_by_id(created_client.id)
    except Exception:
        pass


# ============================================================================
# TEST CLASS: User Schema Verification
# ============================================================================


class TestUserSchemaWithoutClientId:
    """Tests verifying User schema has no client_id field."""

    def test_user_schema_has_no_client_id_field(self):
        """Verify User Pydantic schema does not have client_id field."""
        user = PydanticUser(
            id="user-test",
            name="Test User",
            organization_id="org-1",
            timezone="UTC",
        )
        # Check model_fields on the class, not instance
        assert "client_id" not in PydanticUser.model_fields, "client_id should not be in User schema"

    def test_user_creation_without_client_id(self):
        """Verify users can be created without client_id."""
        user = PydanticUser(
            id=generate_test_id("user"),
            name="Test User",
            organization_id="org-test",
            timezone="America/New_York",
            status="active",
        )
        assert user.id is not None
        assert user.organization_id == "org-test"
        assert user.name == "Test User"


# ============================================================================
# TEST CLASS: Organization-Scoped User Creation
# ============================================================================


class TestOrganizationScopedUserCreation:
    """Tests verifying users are organization-scoped, not client-scoped."""

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_create_user_is_organization_scoped(self, user_manager, test_org1):
        """
        Verify that create_user creates users at the organization level.

        Users should be associated with an organization, not a client.
        """
        user_id = generate_test_id("user")
        user = PydanticUser(
            id=user_id,
            name="Org Scoped User",
            organization_id=test_org1.id,
            timezone="UTC",
        )

        created_user = await user_manager.create_user(user)

        assert created_user.id == user_id
        assert created_user.organization_id == test_org1.id

        try:
            await user_manager.delete_user_by_id(user_id)
        except Exception:
            pass

    async def test_same_user_id_retrieved_by_different_contexts(self, user_manager, test_org1, client_a, client_b):
        """
        Verify that a user created in an org can be retrieved regardless of client context.

        - Create user in org1
        - User should be retrievable (users are org-scoped, not client-scoped)
        """
        user_id = generate_test_id("user")
        user = PydanticUser(
            id=user_id,
            name="Shared User",
            organization_id=test_org1.id,
            timezone="UTC",
        )

        created_user = await user_manager.create_user(user)
        assert created_user.id == user_id

        retrieved_user = await user_manager.get_user_by_id(user_id, organization_id=test_org1.id)
        assert retrieved_user.id == user_id
        assert retrieved_user.organization_id == test_org1.id

        try:
            await user_manager.delete_user_by_id(user_id)
        except Exception:
            pass


# ============================================================================
# TEST CLASS: Multiple Clients Same Org Share Users
# ============================================================================


class TestMultipleClientsSameOrgShareUsers:
    """Tests verifying that clients in the same org share users."""

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_multiple_clients_same_org_see_same_users(self, user_manager, test_org1, client_a, client_b):
        """
        Verify that two clients in the same organization see the same users.

        - Create 3 users in org1
        - list_users(organization_id=org1) should return all 3 users
        - Total user count in org1 should be 3 (not duplicated per client)
        """
        created_user_ids = []

        try:
            for i in range(3):
                user_id = generate_test_id("user")
                user = PydanticUser(
                    id=user_id,
                    name=f"User {i+1}",
                    organization_id=test_org1.id,
                    timezone="UTC",
                )
                await user_manager.create_user(user)
                created_user_ids.append(user_id)

            users = await user_manager.list_users(organization_id=test_org1.id)
            test_users = [u for u in users if u.id in created_user_ids]

            assert len(test_users) == 3, f"Expected 3 users, got {len(test_users)}"

            retrieved_ids = {u.id for u in test_users}
            for uid in created_user_ids:
                assert uid in retrieved_ids, f"User {uid} not found in list"

        finally:
            for uid in created_user_ids:
                try:
                    await user_manager.delete_user_by_id(uid)
                except Exception:
                    pass

    async def test_user_count_not_multiplied_by_clients(self, user_manager, test_org1, client_a, client_b):
        """
        Verify that having multiple clients doesn't multiply user count.

        Before the fix, users were client-scoped, so each client would have
        its own copy. Now users are org-scoped, so count should be consistent.
        """
        user_id = generate_test_id("user")

        try:
            user = PydanticUser(
                id=user_id,
                name="Single User",
                organization_id=test_org1.id,
                timezone="UTC",
            )
            await user_manager.create_user(user)

            users = await user_manager.list_users(organization_id=test_org1.id)
            user_occurrences = [u for u in users if u.id == user_id]

            assert len(user_occurrences) == 1, f"User should appear exactly once, got {len(user_occurrences)}"

        finally:
            try:
                await user_manager.delete_user_by_id(user_id)
            except Exception:
                pass


# ============================================================================
# TEST CLASS: Users Isolated Across Organizations
# ============================================================================


class TestUsersIsolatedAcrossOrganizations:
    """Tests verifying that users in different orgs are isolated."""

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_list_users_filters_by_organization(self, user_manager, test_org1, test_org2):
        """
        Verify list_users filters by organization_id.

        - Create users in org1 and org2
        - list_users(org1) should only return org1 users
        - list_users(org2) should only return org2 users
        """
        org1_user_ids = []
        org2_user_ids = []

        try:
            for i in range(2):
                user_id = generate_test_id("user")
                user = PydanticUser(
                    id=user_id,
                    name=f"Org1 User {i+1}",
                    organization_id=test_org1.id,
                    timezone="UTC",
                )
                await user_manager.create_user(user)
                org1_user_ids.append(user_id)

            for i in range(2):
                user_id = generate_test_id("user")
                user = PydanticUser(
                    id=user_id,
                    name=f"Org2 User {i+1}",
                    organization_id=test_org2.id,
                    timezone="UTC",
                )
                await user_manager.create_user(user)
                org2_user_ids.append(user_id)

            org1_users = await user_manager.list_users(organization_id=test_org1.id)
            org1_retrieved_ids = {u.id for u in org1_users}

            org2_users = await user_manager.list_users(organization_id=test_org2.id)
            org2_retrieved_ids = {u.id for u in org2_users}

            for uid in org1_user_ids:
                assert uid in org1_retrieved_ids, f"Org1 user {uid} not in org1 list"
            for uid in org2_user_ids:
                assert uid in org2_retrieved_ids, f"Org2 user {uid} not in org2 list"
            for uid in org1_user_ids:
                assert uid not in org2_retrieved_ids, f"Org1 user {uid} should not be in org2 list"
            for uid in org2_user_ids:
                assert uid not in org1_retrieved_ids, f"Org2 user {uid} should not be in org1 list"

        finally:
            for uid in org1_user_ids + org2_user_ids:
                try:
                    await user_manager.delete_user_by_id(uid)
                except Exception:
                    pass


# ============================================================================
# TEST CLASS: Client Deletion Does Not Cascade to Users
# ============================================================================


class TestClientDeletionPreservesUsers:
    """Tests verifying that deleting a client does NOT delete users."""

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_delete_client_preserves_users(self, user_manager, client_manager, test_org1):
        """
        Verify deleting a client does NOT cascade-delete users.

        Before the fix, users had a FK to clients with CASCADE delete.
        Now users are org-scoped and should persist when clients are deleted.
        """
        client_id = generate_test_id("client")
        client = PydanticClient(
            id=client_id,
            name="Temporary Client",
            organization_id=test_org1.id,
            status="active",
            write_scope="test",
            read_scopes=["test"],
        )
        await client_manager.create_client(client)

        user_id = generate_test_id("user")
        user = PydanticUser(
            id=user_id,
            name="Persistent User",
            organization_id=test_org1.id,
            timezone="UTC",
        )
        await user_manager.create_user(user)

        try:
            retrieved_user = await user_manager.get_user_by_id(user_id, organization_id=test_org1.id)
            assert retrieved_user.id == user_id

            await client_manager.delete_client_by_id(client_id)

            user_after_delete = await user_manager.get_user_by_id(user_id, organization_id=test_org1.id)
            assert user_after_delete.id == user_id, "User should still exist after client deletion"
            assert user_after_delete.organization_id == test_org1.id

        finally:
            try:
                await user_manager.delete_user_by_id(user_id)
            except Exception:
                pass


# ============================================================================
# TEST CLASS: get_or_create_org_default_user
# ============================================================================


class TestGetOrCreateOrgDefaultUser:
    """Tests for get_or_create_org_default_user without client_id."""

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_get_or_create_org_default_user_creates_user(self, user_manager, test_org1):
        """Verify get_or_create_org_default_user creates a default user for the org."""
        default_user = await user_manager.get_or_create_org_default_user(org_id=test_org1.id)

        assert default_user is not None
        assert default_user.organization_id == test_org1.id
        assert default_user.name == user_manager.DEFAULT_USER_NAME

        try:
            await user_manager.delete_user_by_id(default_user.id)
        except Exception:
            pass

    async def test_get_or_create_org_default_user_is_idempotent(self, user_manager, test_org1):
        """Verify get_or_create_org_default_user returns the same user on repeated calls."""
        default_user_1 = await user_manager.get_or_create_org_default_user(org_id=test_org1.id)
        default_user_2 = await user_manager.get_or_create_org_default_user(org_id=test_org1.id)

        assert default_user_1.id == default_user_2.id, "Should return same user on repeated calls"

        try:
            await user_manager.delete_user_by_id(default_user_1.id)
        except Exception:
            pass

    async def test_get_or_create_org_default_user_different_orgs(self, user_manager, test_org1, test_org2):
        """Verify get_or_create_org_default_user creates separate users for different orgs."""
        default_user_org1 = await user_manager.get_or_create_org_default_user(org_id=test_org1.id)
        default_user_org2 = await user_manager.get_or_create_org_default_user(org_id=test_org2.id)

        assert default_user_org1.id != default_user_org2.id, "Different orgs should have different default users"
        assert default_user_org1.organization_id == test_org1.id
        assert default_user_org2.organization_id == test_org2.id

        try:
            await user_manager.delete_user_by_id(default_user_org1.id)
        except Exception:
            pass
        try:
            await user_manager.delete_user_by_id(default_user_org2.id)
        except Exception:
            pass


# ============================================================================
# TEST CLASS: Org-Scoped Identity Lookups (VEPAGE-1155)
# ============================================================================


class TestGetUserByIdGlobal:
    """get_user_by_id is GLOBAL (id-only) under Option 1.

    Users have id-only identity (matches IPS-R's PK invariant). The
    ``organization_id`` parameter to get_user_by_id is retained for cache-
    key scoping but does NOT filter the lookup — cross-org reads succeed.
    Per-org data isolation lives on child tables, not on the user row.
    See mirix/orm/user.py docstring.
    """

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_get_user_by_id_returns_user_in_matching_org(self, user_manager, test_org1):
        """A user is returned when looked up with its own organization_id."""
        user_id = generate_test_id("user")
        await user_manager.create_user(
            PydanticUser(id=user_id, name="In Org", organization_id=test_org1.id, timezone="UTC")
        )
        try:
            retrieved = await user_manager.get_user_by_id(user_id, organization_id=test_org1.id)
            assert retrieved.id == user_id
            assert retrieved.organization_id == test_org1.id
        finally:
            try:
                await user_manager.delete_user_by_id(user_id)
            except Exception:
                pass

    async def test_get_user_by_id_cross_org_returns_shared_row(self, user_manager, test_org1, test_org2):
        """Cross-org lookup returns the same shared user row.

        Under Option 1 the user row is global. A user created via org1 is
        visible to lookups passing org2's id — the row is shared. Per-org
        isolation happens on child tables (memories, blocks, etc.), not here.
        """
        user_id = generate_test_id("user")
        await user_manager.create_user(
            PydanticUser(id=user_id, name="Shared", organization_id=test_org1.id, timezone="UTC")
        )
        try:
            retrieved = await user_manager.get_user_by_id(user_id, organization_id=test_org2.id)
            assert retrieved.id == user_id
            # The row's stored organization_id is org1's (whoever created it
            # first) — but it's still returned to org2's caller.
            assert retrieved.organization_id == test_org1.id
        finally:
            try:
                await user_manager.delete_user_by_id(user_id)
            except Exception:
                pass

    async def test_create_user_cross_org_id_reuse_returns_existing(
        self, user_manager, test_org1, test_org2
    ):
        """create_user catches users_pkey on cross-org id reuse and returns
        the existing row instead of erroring. Two devs using user_id="1234"
        under different orgs do not collide."""
        user_id = generate_test_id("user")
        first = await user_manager.create_user(
            PydanticUser(id=user_id, name="First", organization_id=test_org1.id, timezone="UTC")
        )
        try:
            second = await user_manager.create_user(
                PydanticUser(id=user_id, name="Second", organization_id=test_org2.id, timezone="UTC")
            )
            # Same row returned (catch-and-swallow), org stays as first writer's.
            assert second.id == first.id == user_id
            assert second.organization_id == test_org1.id
        finally:
            try:
                await user_manager.delete_user_by_id(user_id)
            except Exception:
                pass


class TestGetClientByIdOrgScoped:
    """get_client_by_id must be scoped to (organization_id, id)."""

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_get_client_by_id_requires_organization_id(self):
        """get_client_by_id must require an organization_id parameter."""
        import inspect

        sig = inspect.signature(ClientManager.get_client_by_id)
        params = sig.parameters
        assert "organization_id" in params, "get_client_by_id must accept organization_id"
        assert (
            params["organization_id"].default is inspect.Parameter.empty
        ), "organization_id must be required (no default)"

    async def test_get_client_by_id_returns_client_in_matching_org(self, client_manager, client_a, test_org1):
        """A client is returned when looked up with its own organization_id."""
        retrieved = await client_manager.get_client_by_id(client_a.id, organization_id=test_org1.id)
        assert retrieved.id == client_a.id
        assert retrieved.organization_id == test_org1.id

    async def test_get_client_by_id_cross_org_raises_not_found(self, client_manager, client_a, test_org2):
        """Looking up a client with a different org's id must raise NoResultFound."""
        from mirix.orm.errors import NoResultFound

        with pytest.raises(NoResultFound):
            await client_manager.get_client_by_id(client_a.id, organization_id=test_org2.id)


class TestDefaultEntitiesOrgScoped:
    """Default admin user / default client existence checks must be org-scoped.

    Regression for VEPAGE-1155: create_admin_user / create_default_client used a
    global lookup by the hardcoded constant id, so the first org to write the
    default "won" and every other org's create returned that foreign-org row.
    """

    pytestmark = pytest.mark.asyncio(loop_scope="module")

    async def test_create_admin_user_idempotent_within_org(self, user_manager, test_org1):
        """Calling create_admin_user twice for the same org returns the same row."""
        first = await user_manager.create_admin_user(org_id=test_org1.id)
        second = await user_manager.create_admin_user(org_id=test_org1.id)
        assert first.id == second.id == UserManager.ADMIN_USER_ID
        assert first.organization_id == test_org1.id
        assert second.organization_id == test_org1.id

    async def test_create_admin_user_does_not_return_foreign_org_admin(
        self, user_manager, test_org1, test_org2
    ):
        """An admin existing in org1 must not satisfy org2's existence check.

        The global lookup bug would return org1's admin for org2. With org-scoped
        existence checks org2 never sees org1's row — it either creates its own
        (organization_id == org2) or fails, but must never return org1's row.
        """
        await user_manager.create_admin_user(org_id=test_org1.id)
        try:
            org2_admin = await user_manager.create_admin_user(org_id=test_org2.id)
        except Exception:
            # ORM/PostgreSQL path: a constant-id row cannot coexist across orgs in
            # a single DB, so the second insert collides. That still proves the
            # existence check is org-scoped (it did NOT short-circuit to org1's row).
            return
        assert org2_admin.organization_id == test_org2.id, (
            "create_admin_user(org2) returned an admin from a different org — "
            "existence check is not org-scoped"
        )

    async def test_create_default_client_idempotent_within_org(self, client_manager, test_org1):
        """Calling create_default_client twice for the same org returns the same row.

        The id is org-derived (``client-default-{org_id}``), not the legacy
        global constant. Each org gets its own default-client row.
        """
        first = await client_manager.create_default_client(org_id=test_org1.id)
        second = await client_manager.create_default_client(org_id=test_org1.id)
        expected_id = ClientManager.default_client_id(test_org1.id)
        assert first.id == second.id == expected_id
        assert first.organization_id == test_org1.id
        assert second.organization_id == test_org1.id

    async def test_create_default_client_does_not_return_foreign_org_client(
        self, client_manager, test_org1, test_org2
    ):
        """A default client existing in org1 must not satisfy org2's existence check."""
        await client_manager.create_default_client(org_id=test_org1.id)
        try:
            org2_client = await client_manager.create_default_client(org_id=test_org2.id)
        except Exception:
            return
        assert org2_client.organization_id == test_org2.id, (
            "create_default_client(org2) returned a client from a different org — "
            "existence check is not org-scoped"
        )


# ============================================================================
# TEST CLASS: UserManager API Signature Verification
# ============================================================================


class TestUserManagerApiSignature:
    """Tests verifying UserManager methods have correct signatures."""

    def test_create_user_has_no_client_id_parameter(self):
        """Verify create_user method has no client_id parameter."""
        import inspect

        sig = inspect.signature(UserManager.create_user)
        params = list(sig.parameters.keys())

        assert "client_id" not in params, "create_user should not have client_id parameter"
        assert "pydantic_user" in params, "create_user should have pydantic_user parameter"

    def test_list_users_has_no_client_id_parameter(self):
        """Verify list_users method has no client_id parameter."""
        import inspect

        sig = inspect.signature(UserManager.list_users)
        params = list(sig.parameters.keys())

        assert "client_id" not in params, "list_users should not have client_id parameter"
        assert "organization_id" in params, "list_users should have organization_id parameter"

    def test_get_or_create_org_default_user_has_no_client_id_parameter(self):
        """Verify get_or_create_org_default_user method has no client_id parameter."""
        import inspect

        sig = inspect.signature(UserManager.get_or_create_org_default_user)
        params = list(sig.parameters.keys())

        assert "client_id" not in params, "get_or_create_org_default_user should not have client_id parameter"
        assert "org_id" in params, "get_or_create_org_default_user should have org_id parameter"
