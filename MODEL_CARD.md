# Model Card

## System

This unsupervised portfolio system combines TF-IDF representations of prospect descriptions and campaign text with scaled prospect and deal-history features. UMAP or PCA reduces the text representations; K-Means produces the primary segments, while DBSCAN supplies a density-based comparison. Deterministic rules convert cluster statistics into names and outreach recommendations.

## Intended use

- demonstrate a complete NLP and unsupervised-learning workflow;
- explore synthetic prospect groupings through a CLI or Streamlit interface;
- provide reviewable examples of feature engineering, clustering, profiling, and artifact validation.

## Data and outputs

- 500 deterministic synthetic prospects and 200 deterministic synthetic deal records;
- two free-text fields, four prospect numeric fields, and four aggregated deal-history fields;
- `segments.csv`, `segment_profiles.json`, `visualization.html`, and `report.md`;
- no personal information, employer data, production customer data, or trained model binary.

## Evaluation

There are no ground-truth labels, so this project does not claim classification accuracy. The release gate checks operational properties instead: input validation, deterministic generation, finite feature matrices, complete assignments, non-empty names and strategies, deal-history integration, repeatable PCA artifacts, full UMAP execution, and Streamlit rendering.

## Limitations and safeguards

- Synthetic patterns may be easier to cluster than real prospects.
- Cluster selection by an elbow heuristic is a modeling choice, not proof of business validity.
- DBSCAN is included for comparison and does not validate K-Means.
- Named segments and outreach recommendations are authored rules, not LLM output and not autonomous decisions.
- Real deployment would require data-governance review, feature and distribution monitoring, stakeholder validation, outcome-based evaluation, and human approval of outreach.
- UMAP and other numerical libraries may show small cross-platform floating-point differences even when seeded.

## Runtime boundary

The implementation is CPU-based and cross-platform. Python 3.12 and all direct/transitive dependencies are locked by `uv.lock`; CI reproduces the environment on a public Ubuntu runner.
