# N-LDF post-processing for Google Colab

This analysis uses the completed fold-level CSV and does **not** retrain any
classifier or regenerate synthetic samples.

## Files to upload to Colab

- `nldf_postprocessing.py`
- the final `fold_level_results.csv`
- the matching `dataset_audit.csv`

## Colab commands

```python
from google.colab import files
files.upload()
```

Upload the three files, then run:

```python
!pip install -q numpy pandas scipy matplotlib scikit-posthocs
!python nldf_postprocessing.py \
    --input fold_level_results.csv \
    --audit dataset_audit.csv \
    --output-dir nldf_paper_analysis \
    --bootstrap 5000
```

Download all outputs:

```python
!zip -r nldf_paper_analysis.zip nldf_paper_analysis
files.download("nldf_paper_analysis.zip")
```

## Main outputs

- `overall_completeness.csv`
- `completeness_report.csv`
- `missing_evaluations.csv`
- `duplicate_evaluations.csv`
- `global_summary_corrected.csv`
- `classifier_method_summary.csv`
- `severity_summary_corrected.csv`
- `friedman_tests.csv`
- `nldf_pairwise_holm.csv`
- `all_pairwise_exploratory.csv`
- `aggregation_sensitivity.csv`
- `nldf_diagnostics_by_fold.csv`
- `nldf_diagnostics_by_dataset.csv`
- `nldf_diagnostics_by_severity.csv`
- Nemenyi matrices, figures, and `analysis_manifest.json`

The script never replaces missing performance values with zero. Inferential
tests use only datasets having all methods required by the corresponding test.