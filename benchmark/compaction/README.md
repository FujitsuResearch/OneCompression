# Context compaction benchmark

This benchmark is deterministic, data-free, and CPU-only. It creates an agent
trajectory containing large build logs, repeated outputs, Unicode text, and
verification needles. It then measures the production `formal-v1` profile at
the same boundaries used by a host integration:

- occurrence-addressed commit and retry behavior,
- real transcript replacement,
- final provider-request budget enforcement,
- complete catalog discovery,
- exact chunk reconstruction and checksum verification,
- sequential chunk/range service latency and indexed literal-search accuracy,
- compaction ratio and operation latency.

Run from the repository root:

```bash
python benchmark/compaction/run_benchmark.py \
  --output-dir results/compaction/formal-v1
```

The output directory contains `benchmark-results.json`, a body-free
`run-report.json`, and a single-run dashboard in PNG and SVG. Raw observation
bodies remain only in the private SQLite archive and are never copied to the
reports or figures.

Use `--observations`, `--body-chars`, `--context-chars`, `--keep-recent`, and
`--seed` to scale the run. The default `--keep-recent 0` makes even a single
observation larger than the provider envelope compactable. Set it to a positive
number when measuring a host policy that preserves recent turns verbatim; the
context limit must then fit those turns. Compare only reports with identical
corpus parameters and OneComp versions.

For a scale-oriented CPU run that stresses linear summary construction,
partial recall, and catalog/search growth:

```bash
python benchmark/compaction/run_benchmark.py \
  --output-dir results/compaction/scale \
  --observations 8 --body-chars 1600000 --context-chars 32000
```

The report includes full sequential chunk reconstruction time and exactness,
so repeating this command with increasing `--body-chars` exposes super-linear
partial-recall regressions rather than measuring only the first few chunks.

## Coding-agent qualification

The public benchmark measures the compaction mechanism without requiring a
model account. The deterministic multi-agent contract test additionally runs
three concurrent coding roles, compacts each independent history, reconstructs
an archived result from bounded chunks, and checks token/tenant isolation:

```bash
pytest -q tests/onecomp/mcp/test_multi_agent_coding_workflow.py
```

For an authenticated coding agent, initialize a project-local archive and put
verbose build/test commands behind the standalone boundary:

```bash
onecomp-arc --config .onecomp/compaction/config.json init
onecomp-arc --config .onecomp/compaction/config.json run -- pytest -vv
```

Live-agent quality depends on the selected model, host version, credentials,
task corpus, and provider limits, so it is a separate qualification step rather
than a network-dependent CI assertion. A valid host qualification must verify
task success, preserved child exit codes, bounded model-visible output, archive
checksums, and exact recall; compression ratio alone is insufficient.
