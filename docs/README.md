# Documentation

Absolute paths in these documents were anonymised for the release: workspace paths are relative, and cluster
locations are replaced by placeholders (`<PDB_MIRROR>`, `<AFDB_DIR>`, `<SHARED_WORK_FS>`, `<SCRATCH>`, `<ENV_STORE>`,
`<KIT_SOURCE_DIR>`, `<GPU_NODE_1>`/`<GPU_NODE_2>`, `<user>`). Nothing else in the text was changed.

| file | content |
|---|---|
| `REPORT.md` | final report: summary, approach, val and test results, limitations, reproducibility |
| `RESEARCH_TREE.md` | every experiment tried and its outcome, including negative arms |
| `DESIGN.md` | study design, pre-registration and test-set rules |
| `LITERATURE.md` | literature survey |
| `LESSONS.md` | process rules the agents adopted during the project |
| `DATA.md` | dataset provenance (same as `../DATA.md`) |
| `reviews/` | independent agent reviews: design panels (round 1/2), report fact-checks, lessons retrospectives |
| `reports/` | per-task summaries written by the sub-agents (engine, scoring fit, AF mode arms, final evaluation, verification, ...); they document the agent process |

The documents were written in the research workspace, so the paths they mention follow that layout. In this
repository:

- `results/final/...` → `predictions/...`
- `results/splits/test_*` → `predictions/splits/`
- `reports/`, `reviews/` → `docs/reports/`, `docs/reviews/`

Many scripts, configs and result folders that the documents cite (negative arms, co-folding, baselines, val sweeps)
are not part of this release (see `../README.md`, "Not included").
