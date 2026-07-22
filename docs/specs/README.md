# Work-item specs (the-loop 3-phase model)

> Per-work-item Kiro-style specs live here, one directory per work item:
> `docs/specs/<id>/` where `<id>` is the ticket id (e.g. `VEPAGE-42`, `issue-1`).
> Path configurable via `workflow.specDir` in `.sdlc/config.yaml`.

Each work item is specified in three human-reviewed phases, then executed:

```
docs/specs/<id>/
├── requirements.md   # or bugfix.md for the minimal bug path (phase: requirements-definition)
├── design.md         # phase: design
├── tasks.md          # DAG of TDD tasks (phase: tasks-breakdown)
└── execution-log.md  # self-checking progress ledger (implementation → complete)
```

Templates are instantiated per work item from `.sdlc/templates/`
(`requirements.md`, `bugfix.md`, `design.md`, `tasks.md`, `execution-log.md`).

## Rules

- **Every work item has a ticket** (Jira or GitHub Issue, per `ticketing.system`).
- **Human review per phase** — a phase is not started until the previous one is
  `approved` (front-matter `status:` and `approvedBy:`), when
  `workflow.requireHumanReviewPerPhase` is true.
- **Single source of truth** — the ticket *references* these files by path; it never
  duplicates them. Subsequent changes are edits to the docs, not new ticket comments.
- **Phase tags** — the work item's phase is mirrored as a ticket label
  `<phaseLabelPrefix><phase>` (e.g. `loop:design`).
- **Paper trail** — every decision/opinion taken from a human lands as a comment on
  the ticket or PR; messaging channels only *signal* that attention is needed.

## Relationship to the classic pipeline

`docs/projects/<slug>/` holds the classic pipeline's artifacts (plan / architecture /
story specs). the-loop work items use `docs/specs/<id>/` instead. Both coexist;
`docs/architecture/architecture.md` and `docs/decisions/` are shared by both.
