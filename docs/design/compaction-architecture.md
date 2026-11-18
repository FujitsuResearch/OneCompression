# Formal compaction architecture

The production path is split into narrow layers so storage guarantees,
model-facing bounds, and host behavior can be tested independently.

```text
host tool event
  -> CompactionRunner.record_observation(source_key=...)
  -> ArcCoreService
  -> SQLiteObservationArchive transaction
  -> observation ID + bounded citation

standalone coding command
  -> onecomp-arc run -- argv (no shell)
  -> captured stdout/stderr + child status
  -> durable observation commit
  -> citation only, or fail-open original bytes

host transcript
  -> CompactionRunner.enforce_char_budget / enforce_token_budget
  -> actual host-list replacement
  -> provider serialization
  -> InvocationBudgetGuard
  -> provider call

model action
  -> strict parser + per-response attempt allowance
  -> catalog / metadata / full / chunk / range service
  -> bounded exact response
```

## Identity and commit

The ID contains a fixed-width occurrence sequence and a fixed-width BLAKE2s
fingerprint. The sequence, body, checksum, metadata, and next-sequence value
commit in one `BEGIN IMMEDIATE` transaction. A citation is returned only after
commit. Different occurrences never collapse merely because their bodies are
equal.

`source_key` enforces exactly-once ingestion at the host boundary. A matching
retry returns its existing record. Different command, body, return code,
tags, or extras under that key raises `SourceOccurrenceConflict`. When a host
has no stable occurrence ID, the adapter omits this key and preserves repeated
equal results as separate occurrences; content equality is never used as an
exactly-once signal.

## Recovery and budgets

Every body has a SHA-256 and exact UTF-8 byte/character counts. Full recall is
fail-closed when it exceeds the configured unit envelope. Text chunks are
adjacent character ranges constructed under the same counter. Token chunks
retain the codec's original token IDs, rather than relying on decode/re-encode
equivalence.

Character ranges use fixed character blocks with sparse UTF-8 byte offsets and
per-block hashes. Incremental SQLite BLOB reads fetch and verify only overlapping
blocks without storing a second body copy. Existing archives are fully verified
and backfilled on first range access; damaged acceleration metadata is rebuilt
only after canonical verification. Default character chunks derive spans
arithmetically; custom counters cache body-free spans. Recall telemetry updates
are buffered and flushed transactionally, avoiding one fsync per chunk without
weakening the body commit guarantee.

Hook recovery commands open committed archives in SQLite read-only mode. This
keeps exact recall available inside agent sandboxes that cannot chmod, migrate,
or write telemetry in the hook-owned cache. Writable library and host-runner
paths retain batched recall telemetry and transparent sparse-index backfill.

Summary-cap accounting tracks lengths as integers and assembles only the final
candidate, making it linear in trajectory size. Formal token enforcement uses
the host's authoritative counter over sanitized provider messages and fails
closed when the immutable prefix plus live tail cannot fit.

The provider guard counts the final request object immediately before calling
the provider. Audit counters distinguish checked, dispatched, and rejected
requests. A host cannot claim complete budget coverage from `check()` calls
alone.

## Security and privacy

Persistent directories and database files are created with private modes when
supported; SQLite database permissions are restricted before the first
connection. Every API that returns an archived body verifies its identity,
sequence, and checksum first. MCP sessions use random secret tokens stored as
hashes, tenant binding, TTL refresh, capacity limits, and a lease that
serializes operations against close. A formal persistent archive stores a
hashed owner binding and has at most one live session per server process.
Model-supplied archive paths are confined beneath the server's configured root.

Telemetry stores IDs, counts, return codes, operation kinds, and sizes. It does
not store body text. Analyzer reports additionally exclude free-form `extra`
metadata because host payloads may place sensitive data there.

Host guarantees are capability-specific. Claude Code and OpenCode expose
output-replacement boundaries, current Codex replaces completed output through
PostToolUse block feedback, and OpenHands exposes native condensation. The
standalone command boundary remains the version-independent fallback. MCP
storage alone never implies ownership of a client's private transcript.

Encryption, retention, and secure deletion are infrastructure policies rather
than implicit archive behavior. This keeps the core dependency-free and avoids
claiming guarantees the process cannot enforce on every filesystem.

## Compatibility

The older `ArcEngine` remains as `experimental-v1`. New standalone,
authenticated MCP, hook, provider-boundary, and analyzer integrations use
`CompactionRunner` with `formal-v1`. JSONL migration imports each recoverable
legacy row as an occurrence and records its old ID as an alias.
