# Verification Record

Verified locally on August 4, 2026.

## Host

- macOS 26.6;
- Apple M3 Max, 40-core GPU;
- ARM64;
- Metal 4;
- Python 3.12.13.

## Result

The Metal preflight passed and all 3 unit tests passed. The full pipeline generated 500 prospects and 200 deal-history rows, extracted two 500-term TF-IDF representations, reduced each to 10 UMAP components, combined them with 4 scaled commercial features, and clustered the resulting 24-feature matrix.

Automatic selection chose five K-Means clusters and produced five named audience profiles with outreach recommendations. DBSCAN ran as a comparison and exposed 30 dense groups plus 280 noise points. The CLI wrote 500 segment assignments, JSON profiles, an interactive HTML visualization, and a Markdown report.

## Reproduce

```bash
./scripts/verify_macos_metal.sh
python -m unittest discover -s tests -v
python generate_data.py
python segment.py --input data/prospects.csv --clusters auto --output results
```
