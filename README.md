# Cross-lingual prior-art recovery pilot

This repository contains the executable core of the pilot described in
[`PILOT_SPEC.md`](./PILOT_SPEC.md).

## Quick start

The demo is deliberately offline and uses a small synthetic corpus. It is a
pipeline smoke test, not research evidence:

```powershell
python scripts\run_pilot.py --demo --output-dir results\demo
```

The command writes `corpus.jsonl`, `queries.jsonl`, `metrics.csv`,
`failure_cases.md`, and `report.md`. The implementation uses only the Python
standard library, so it can be run before choosing an LLM, embedding model,
or ACL data source.

For real data, provide a JSONL corpus with the fields documented in
`src/pilot/corpus.py` and run:

```powershell
python scripts\run_pilot.py --corpus data\processed\corpus.jsonl --output-dir results\run
```

The current executable scope is the P0/P1 retrieval and evaluation core.
ACL discovery/download, PDF extraction, LLM facet generation/judging, dense
retrieval, and human-query import are intentionally separate integration
surfaces and are not silently replaced by the demo fixture.

