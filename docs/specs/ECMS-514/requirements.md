---
type: requirements
phase: requirements-definition
workItem: "ECMS-514"
status: approved
approvedBy: ["auto-waived: 1-point P1 with root cause pre-traced by SRF (Calvin Hu) and a single narrow goal stated on the ticket; review plan set at kickoff by lparzych"]
collaborators: []
overrides: {}
---

# Requirements: Send LXS transcript payload as a single content block

> Phase 1 of 3 (requirements → design → tasks). Kiro-style spec
> (https://kiro.dev/docs/specs/), but written to be READ — see the authoring rules
> below. Reviewed per the story's review plan (Phase Gates rule in the workspace
> CLAUDE.md); for most stories this document is a hard human gate.

<!-- AUTHORING RULES (keep this comment; reviewers use it to judge the doc)
1. TELL A STORY. Establish what's true today, what hurts, and why we're changing it —
   before any requirement appears. Name real files, behaviors, and evidence so the
   reader learns something about the codebase and can CHECK the reasoning. AI-authored
   claims are hypotheses until a human reviews them; that's the point of the gate.
2. LENGTH IS PROPORTIONAL TO THE CHANGE. A small ticket reads in a page. Don't pad.
3. EVERY REQUIREMENT EARNS ITS PLACE. Each one opens with 1–3 sentences of setup and
   justification tying it back to the story. Bare EARS clauses with no reasoning train
   reviewers to skip them — which defeats the review.
4. The reviewer experience to aim for: a senior dev reads top to bottom without
   drudgery, spots any wrong reasoning, and finishes knowing the subsystem better.
-->

## The story

On the save path, the queue worker packs a multi-turn conversation into the agent input
via `flatten_messages_for_agent` (`mirix/utils.py:1909`, called from
`mirix/queue/worker.py:357`). For every turn it appends **two** separate
`{"type": "text"}` dicts — a `[USER]`/`[ASSISTANT]` marker, then the turn's content —
and `convert_message_to_mirix_message` converts each dict into its own `TextContent`
part on a single `MessageCreate`. A long transcript therefore reaches the LLM request
as ~2N tiny text content parts rather than one flattened string.

Downstream, GenOS SRF sends **each content block** of an LXS request to its scanning
detectors. The GenOS SRF team (Calvin Hu) observed ECMS requests for experience id
`5721b358-7e56-4ebb-bc1a-1cd16c5ea681` fanning out up to ~2000 tiny parts per request —
mostly single-line blocks like `[USER]` — overloading their detectors. They have
temporarily disabled some detectors for our expid until this is fixed, which is why the
ticket is P1. Sample tid from SRF: `1-6a5908b4-307575a25fa558a069910cc4`; source thread:
https://intuit.enterprise.slack.com/archives/C0A8L59QAKD/p1784223485188039

The multi-part shape is not intentional design: the docstring of
`flatten_messages_for_agent` records that it deliberately reproduced the old pre-queue
flattening from `rest_api.py` so the ECMS-73 refactor wouldn't change save-path
semantics. The semantics worth preserving are the *text* the model sees (the
`[USER]`/`[ASSISTANT]` delineation), not the accidental part-per-line packaging.

Ticket: [ECMS-514](https://jira.cloud.intuit.com/browse/ECMS-514)

## The goal

The save-path LLM request to LXS carries the entire flattened transcript as a
**single** text content block in the UserMessage, preserving the existing
`[USER]`/`[ASSISTANT]` delineation within the text. This eliminates the
multi-content-part fanout so SRF can re-enable its detectors. Joining at the source
(`flatten_messages_for_agent`) is the right shape because it fixes every consumer of
the packed message in one place, rather than coalescing parts later in each LLM-client
serializer.

## Out of scope

- The legacy `input_messages` wire-format fallback in `worker.py` — because those
  messages were packed by pre-migration producers and the field is being retired.
- Changing the `[USER]`/`[ASSISTANT]` marker convention or the "non-user roles are
  marked `[ASSISTANT]`" ternary — because this story fixes packaging, not prompt
  semantics.
- Any ECMS-repo (`context-and-memory-service`) change — the flattening lives entirely
  in MIRIX.

## Requirements

### R1 — Single text block for flattened transcripts

This is the whole story: the fanout exists because each marker and each turn's text
becomes its own `TextContent`.

**User story:** As the GenOS SRF platform, I want each ECMS save-path LXS request to
carry the transcript as one text content block, so that per-block detector scanning is
not overloaded and detectors can be re-enabled for the ECMS expid.

**Acceptance criteria (EARS):**

1. WHEN `flatten_messages_for_agent` receives turns whose contents are all strings
   THEN the system SHALL return a single `MessageCreate` whose content contains
   exactly one `TextContent` part holding the full flattened transcript.
2. WHEN the transcript is flattened THEN the system SHALL preserve the existing
   per-turn `[USER]`/`[ASSISTANT]` markers and turn ordering within the single text
   block, and SHALL preserve the existing role ternary (any non-`user` role is marked
   `[ASSISTANT]`).

### R2 — Non-text content parts are not dropped

Turn content may legally be a list containing non-text items (e.g. image parts); today
those pass through as their own content parts. Collapsing must not silently discard
them.

**User story:** As a MIRIX save-path consumer, I want non-text content items to survive
flattening unchanged, so that multimodal turns keep working.

**Acceptance criteria (EARS):**

1. IF a turn's content list contains non-text items THEN the system SHALL keep those
   items as separate content parts while still coalescing all text (markers and text
   items) into the minimal number of text blocks around them.
2. WHEN all content is textual THEN the system SHALL emit no additional content parts
   beyond the single text block (R1.1).

## Non-functional requirements

- **Observability parity:** no new logging required; the change must not alter
  LangFuse span structure on the save path — the packed message is inspected in
  existing traces and should simply show one text block.

## Open questions

None — the fix location and desired shape were established in the SRF source thread.
