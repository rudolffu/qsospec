# Fast resume benchmark

Date: 2026-09-03. Environment: local macOS conda base, Python 3.12.

The pre-change production observation was approximately 5.5 minutes to scan
and skip one completed shard. That path projected spectral vectors and built a
`SpectrumData` object for every completed row before testing its object key.

`benchmarks/benchmark_fast_resume.py` exercises a production-shaped synthetic
Parquet shard with 224 rows, 4,000 pixels per row, seven row groups, explicit
object keys/input row indices/shard IDs, and deterministic schema-v5 product
paths. It runs no numerical fit. Empty placeholder product files are used only
to measure the deterministic membership lookup; strict completion-marker
creation separately reconciles real Parquet object-key columns.

| Case | Wall time | Vector rows loaded | Vector rows avoided | Workers | Fits |
|---|---:|---:|---:|---:|---:|
| Fully complete scalar plan | 0.0101 s | 0 | 224 | 0 | 0 |
| 221 complete, 3 unfinished: plan | 0.0089 s | 0 | 221 | 0 | 0 |
| 221 complete, 3 unfinished: sparse vector read | 0.0175 s | 3 | 221 | 0 | 0 |
| Fresh 224-row scalar plan | 0.0091 s | 0 | 0 | 0 | 0 |

For the fully complete case, the identity scan took 0.0045 s and direct
run-store membership checks took 0.0053 s. The partial reader returned exactly
physical rows 221, 222, and 223. Lightweight and full scanners produced equal
ordered `SpectrumInput` descriptors. The automated suite also compares fresh
optimized and established batch products through the existing run-store tests.

The synthetic timing is not a substitute for the agnkiaa filesystem
measurement. After deployment, run the MLSpecZ `resume-shards` stage once to
bootstrap strict `SHARD_COMPLETE.json` files and record its log. Expected
behavior is:

- a valid marker: scalar membership/identity and metadata validation, no
  `fit_batch`, vectors, workers, or fits;
- a complete shard without a marker: scalar planning plus one strict
  object-key reconciliation, then atomic marker creation;
- a partial shard: vector loading and fitting proportional only to unfinished
  and retried failures;
- a fresh shard: the same scientific fit, preceded by a small scalar scan.

Reproduce the synthetic benchmark with:

```bash
PYTHONPATH=src /Users/yuming/miniforge3/bin/python \
  benchmarks/benchmark_fast_resume.py
```
