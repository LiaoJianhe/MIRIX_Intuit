"""Result dataclasses for relational-provider named queries.

When a named query's SELECT projection doesn't correspond to a full entity row
(scalar aggregates, single-column projections, custom joined shapes), callers
must pass a ``result_set_entity_class`` to ``find_using_named_query`` so the
provider knows how to hydrate the response. The classes below cover the common
shapes used by the manager layer.

Lives in MIRIX (not the host application) because every caller is a MIRIX
manager. Host applications that wrap MIRIX register their own relational
provider implementation; that provider consumes these classes via the same
public ``result_set_entity_class`` kwarg.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IdOnlyResult:
    """For ``SELECT id FROM ...`` named queries (e.g. resolve-id lookups)."""

    id: Optional[str] = None


@dataclass
class CountResult:
    """For ``SELECT COUNT(*) AS count FROM ...`` named queries (size, dashboard counters)."""

    count: Optional[int] = 0


@dataclass
class FileStatsResult:
    """For ``file_manager.get_file_stats``.

    Projection: ``COUNT(id) AS total_files, SUM(file_size) AS total_size,
    COUNT(DISTINCT file_type) AS unique_types``.
    """

    total_files: Optional[int] = 0
    total_size: Optional[int] = 0
    unique_types: Optional[int] = 0


@dataclass
class MaxOccurredAtResult:
    """For ``memory_citation_manager.max_occurred_at_for_memory``.

    Projection: ``MAX(occurredAt) AS max_occurred_at``. Single-column aggregate
    with no ``id`` field — must be paired with ``skip_entity_mapping=True`` so
    the underlying provider doesn't try to bind the row to the memoryCitations
    entity schema. Used by the temporal guard to detect out-of-order writes.
    """

    max_occurred_at: Optional[object] = None
