# Notebook workflow

Launch the executable tutorial from this repository root:

```bash
uv sync --locked --group notebook
uv run --group notebook jupyter lab notebooks/01_data_to_validation.ipynb
```

The notebook generates disorganized prospect and deal-history exports plus their cleaned counterparts under `notebooks/data/`, runs the real TF-IDF/UMAP/K-Means/DBSCAN pipeline under `notebooks/artifacts/`, and executes the unit tests.
