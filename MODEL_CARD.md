# Model Card

## System

An unsupervised audience-segmentation pipeline combines TF-IDF representations of descriptions and campaign text with scaled commercial features, reduces the feature space with UMAP or PCA, and compares K-Means with DBSCAN. It then converts clusters into descriptive names and rule-based outreach recommendations.

## Data and outputs

- 500 synthetic prospects and 200 synthetic deal-history records;
- text, industry, revenue, employee, social-presence, and advertising-spend inputs;
- `segments.csv`, `segment_profiles.json`, an interactive visualization, and a Markdown report;
- CLI and Streamlit interfaces use the same orchestration path.

## Hardware boundary

Tests are gated to ARM64 macOS hosts with Metal support. TF-IDF, UMAP, K-Means, and DBSCAN are CPU workloads on Apple Silicon; this repository does not claim that these scikit-learn pipelines use Metal.

## Limitations

The segment labels and outreach text are deterministic presentation rules applied to synthetic patterns. They require human review before use, and the clusters should be revalidated when the source population or feature distribution changes.
