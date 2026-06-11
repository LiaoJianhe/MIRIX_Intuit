"""
Relational and search provider registry tests for Mirix (providers).

Usage:
    pytest tests/test_provider_registration.py -v
"""

import pytest

from mirix.database.relational_provider import (
    get_registered_relational_providers,
    get_relational_provider,
    register_relational_provider,
    unregister_relational_provider,
)
from mirix.database.search_provider import (
    get_registered_search_providers,
    get_search_provider,
    register_search_provider,
    unregister_search_provider,
)


@pytest.fixture(autouse=True)
def cleanup_relational_registry():
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    yield
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)


@pytest.fixture(autouse=True)
def cleanup_search_registry():
    for name in list(get_registered_search_providers().keys()):
        unregister_search_provider(name)
    yield
    for name in list(get_registered_search_providers().keys()):
        unregister_search_provider(name)


class TestRelationalProviderRegistry:
    def test_register_stores_provider(self):
        p = object()
        register_relational_provider("ips_relational", p)
        assert get_registered_relational_providers()["ips_relational"] is p

    def test_get_returns_active_provider(self):
        p = object()
        register_relational_provider("ips_relational", p)
        assert get_relational_provider() is p

    def test_unregister_removes_provider(self):
        p = object()
        register_relational_provider("ips_relational", p)
        unregister_relational_provider("ips_relational")
        assert "ips_relational" not in get_registered_relational_providers()
        assert get_relational_provider() is None

    def test_get_registered_returns_all_providers(self):
        a, b = object(), object()
        register_relational_provider("first", a)
        register_relational_provider("second", b)
        reg = get_registered_relational_providers()
        assert reg == {"first": a, "second": b}
        assert reg is not get_registered_relational_providers()


class TestSearchProviderRegistry:
    def test_register_stores_provider(self):
        p = object()
        register_search_provider("ips_search", p)
        assert get_registered_search_providers()["ips_search"] is p

    def test_get_returns_active_provider(self):
        p = object()
        register_search_provider("ips_search", p)
        assert get_search_provider() is p

    def test_unregister_removes_provider(self):
        p = object()
        register_search_provider("ips_search", p)
        unregister_search_provider("ips_search")
        assert "ips_search" not in get_registered_search_providers()
        assert get_search_provider() is None

    def test_get_registered_returns_all_providers(self):
        a, b = object(), object()
        register_search_provider("first", a)
        register_search_provider("second", b)
        reg = get_registered_search_providers()
        assert reg == {"first": a, "second": b}
        assert reg is not get_registered_search_providers()


class TestRegistryLastWins:
    def test_multiple_relational_only_last_active(self):
        p1, p2 = object(), object()
        register_relational_provider("first", p1)
        register_relational_provider("second", p2)
        assert get_relational_provider() is p2
        assert get_registered_relational_providers() == {"first": p1, "second": p2}

    def test_multiple_search_only_last_active(self):
        p1, p2 = object(), object()
        register_search_provider("first", p1)
        register_search_provider("second", p2)
        assert get_search_provider() is p2
        assert get_registered_search_providers() == {"first": p1, "second": p2}


class TestEmptyRegistry:
    def test_get_relational_provider_none_when_empty(self):
        assert get_relational_provider() is None

    def test_get_search_provider_none_when_empty(self):
        assert get_search_provider() is None
