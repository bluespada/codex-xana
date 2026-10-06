# TUI bottom pane (state machines)

When changing the paste-burst or chat-composer state machines in this folder, keep the docs in sync:

- Update the relevant module docs (`chat_composer.rs` and/or `paste_burst.rs`) so they remain a
  readable, top-down explanation of the current behavior.
- Keep implementations/docstrings aligned unless a divergence is intentional and documented.

Practical check:

- After edits, sanity-check that docs mention only APIs/behavior that exist in code (especially the
  Enter/newline paths and `disable_paste_burst` semantics).

## Note for future deletions (fork)

The multi-provider work in this fork gives `codex-rs/codex-providers` its own
protocol modules built on vendor SDK vocabularies (`async-openai` for Chat
Completions, `claudius` for the Messages API, and a Responses module on
`async-openai` still to come). Nothing routes to a Responses module yet, and it
stays that way until the Responses tests in `codex-rs/core` are the gate.

That leaves the Responses machinery codex uses for its own backend orphaned
rather than wrong:

- `codex-rs/codex-api/src/endpoint/responses.rs` and `codex-rs/codex-api/src/sse/responses.rs`
- the Responses request body, stream decoding, and `x-openai-*` header handling in `codex-rs/core/src/client.rs`

Do not delete any of it as a side effect of unrelated work. It can only go once
core sends Responses traffic through `codex-providers`, and at that point the
guardian review metadata, zstd request compression, and rollout inference trace
that live there have to be rehomed or dropped deliberately.
