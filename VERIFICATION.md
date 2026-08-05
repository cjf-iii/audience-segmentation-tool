# Verification Record

Recorded August 4, 2026.

## Release gate

```bash
uv sync --locked --group dev --group notebook
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run --group notebook jupyter nbconvert --to notebook --execute notebooks/01_data_to_validation.ipynb --stdout > /dev/null
uv run python generate_data.py
uv run python segment.py --input data/prospects.csv --deals data/deal_history.csv --clusters auto --output results
uv run python scripts/verify_artifacts.py
```

The Streamlit interface is also loaded with `streamlit.testing.v1.AppTest` and must complete without an application exception.

## Acceptance criteria

- one locked `uv` environment on Python 3.12;
- public CPU-based GitHub Actions workflow;
- deterministic 500-row prospect and 200-row deal generators;
- every reviewer-notebook cell executes in the locked notebook environment;
- deal-history aggregates present in the clustering matrix and final CSV;
- finite feature matrix and one assignment per prospect;
- all four documented artifacts generated;
- repeatable SHA-256 hashes for two independent PCA runs;
- full UMAP and DBSCAN path completes;
- artifact verifier and Streamlit smoke test pass.

## Recorded result

All 5 tests passed. Every reviewer-notebook cell executed successfully. The integration test executed two independent PCA runs and produced identical SHA-256 hashes for every documented artifact.

The full UMAP path then generated 500 prospects and 200 deals, merged all 200 deal rows into four per-company features, and produced a 500 × 28 clustering matrix. Automatic elbow selection chose 5 K-Means clusters. The DBSCAN comparison found 14 dense clusters and marked 407 prospects as noise. The artifact verifier confirmed 500 assignments, 5 named K-Means profiles, projection coordinates, and the deterministic HTML element. Streamlit's application test rendered the generated results without an exception.

The seeded UMAP run emitted its expected notice that deterministic execution constrains `n_jobs` to one; this is a reproducibility choice, not a failure.
