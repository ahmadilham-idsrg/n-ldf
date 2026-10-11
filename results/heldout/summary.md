# Held-out source evaluation

Protocol SHA-256: `7fe71cc3d36bb935bbd5684ecf508e310641c3d54edbe08d823401b8c1250bd9`

13 tasks, 10 sources, 13,182 fit records.

## Source-level means

| Method       |     F1 |   Precision |   Recall |   Specificity |   G-Mean |   AUC-ROC |   AUPRC |
|:-------------|-------:|------------:|---------:|--------------:|---------:|----------:|--------:|
| ADASYN       | 0.7249 |      0.7243 |   0.7864 |        0.9699 |   0.8393 |    0.9574 |  0.7913 |
| B-SMOTE      | 0.725  |      0.7259 |   0.7818 |        0.9714 |   0.8364 |    0.9569 |  0.7934 |
| Baseline     | 0.6244 |      0.6945 |   0.6002 |        0.9906 |   0.6682 |    0.9403 |  0.7795 |
| Baseline-thr | 0.6864 |      0.6803 |   0.7697 |        0.9443 |   0.8031 |    0.9402 |  0.7793 |
| N-LDF        | 0.6644 |      0.7088 |   0.663  |        0.9873 |   0.7202 |    0.945  |  0.7833 |
| N-LDF-r1.0   | 0.6662 |      0.7099 |   0.6662 |        0.9869 |   0.7232 |    0.9441 |  0.7818 |
| ROS-r0.75    | 0.7117 |      0.7149 |   0.7696 |        0.9716 |   0.827  |    0.9557 |  0.788  |
| ROS-r1.0     | 0.7036 |      0.6987 |   0.7723 |        0.9688 |   0.824  |    0.954  |  0.7861 |
| SMOTE        | 0.73   |      0.731  |   0.787  |        0.9727 |   0.8418 |    0.9574 |  0.7924 |
| SMOTE-Tomek  | 0.7283 |      0.729  |   0.7848 |        0.9728 |   0.8399 |    0.9566 |  0.7912 |
| SMOTE-r0.75  | 0.7293 |      0.7318 |   0.7803 |        0.9745 |   0.8372 |    0.9578 |  0.7929 |

Friedman (six archived strategies, source-level F1): chi2 = 15.04, p = 0.0102; five oversamplers: chi2 = 8.06, p = 0.0893.

Mean F1 rank (source level): SMOTE 2.25, SMOTE-Tomek 2.60, B-SMOTE 3.45, ADASYN 3.65, N-LDF 3.95, Baseline 5.10

## Family `archived`

| metric | contrast (a − b) | mean diff | W–T–L (tasks) | source Holm p | task Holm p | cluster 95% interval |
|---|---|---:|---:|---:|---:|---:|
| F1 | N-LDF − Baseline | +0.0340 | 10–1–2 | 0.078 | 0.046 | [+0.006, +0.073] |
| F1 | N-LDF − SMOTE | -0.0621 | 3–0–10 | 0.336 | 0.321 | [-0.124, -0.008] |
| F1 | N-LDF − B-SMOTE | -0.0575 | 4–0–9 | 0.387 | 0.503 | [-0.120, -0.002] |
| F1 | N-LDF − ADASYN | -0.0567 | 5–0–8 | 0.387 | 0.503 | [-0.121, +0.000] |
| F1 | N-LDF − SMOTE-Tomek | -0.0603 | 4–0–9 | 0.336 | 0.503 | [-0.122, -0.006] |
| AUPRC | N-LDF − B-SMOTE | -0.0088 | 6–0–7 | 1.000 | 1.000 | [-0.026, +0.003] |
| AUPRC | N-LDF − Baseline | +0.0015 | 6–0–7 | 1.000 | 1.000 | [-0.007, +0.012] |
| AUPRC | N-LDF − ADASYN | -0.0077 | 7–1–5 | 1.000 | 1.000 | [-0.026, +0.008] |
| AUPRC | N-LDF − SMOTE-Tomek | -0.0070 | 8–1–4 | 1.000 | 1.000 | [-0.024, +0.006] |
| AUPRC | N-LDF − SMOTE | -0.0078 | 7–1–5 | 1.000 | 1.000 | [-0.025, +0.005] |

## Family `amount`

| metric | contrast (a − b) | mean diff | W–T–L (tasks) | source Holm p | task Holm p | cluster 95% interval |
|---|---|---:|---:|---:|---:|---:|
| F1 | ROS-r1.0 − ROS-r0.75 | -0.0068 | 3–1–9 | 0.223 | 0.130 | [-0.015, -0.001] |
| F1 | SMOTE − SMOTE-r0.75 | -0.0001 | 6–1–6 | 1.000 | 1.000 | [-0.005, +0.006] |
| F1 | N-LDF-r1.0 − N-LDF | +0.0009 | 5–1–7 | 1.000 | 1.000 | [-0.001, +0.004] |
| AUPRC | SMOTE − SMOTE-r0.75 | -0.0012 | 3–1–9 | 1.000 | 0.470 | [-0.004, +0.002] |
| AUPRC | ROS-r1.0 − ROS-r0.75 | -0.0020 | 3–1–9 | 1.000 | 0.470 | [-0.006, +0.002] |
| AUPRC | N-LDF-r1.0 − N-LDF | -0.0014 | 5–2–6 | 1.000 | 0.563 | [-0.004, +0.000] |

## Family `mechanism_r0.75`

| metric | contrast (a − b) | mean diff | W–T–L (tasks) | source Holm p | task Holm p | cluster 95% interval |
|---|---|---:|---:|---:|---:|---:|
| F1 | N-LDF − SMOTE-r0.75 | -0.0622 | 3–0–10 | 0.211 | 0.119 | [-0.122, -0.010] |
| F1 | SMOTE-r0.75 − ROS-r0.75 | +0.0137 | 9–1–3 | 0.164 | 0.119 | [+0.002, +0.031] |
| F1 | N-LDF − ROS-r0.75 | -0.0485 | 5–0–8 | 0.432 | 0.340 | [-0.112, +0.010] |
| AUPRC | SMOTE-r0.75 − ROS-r0.75 | +0.0039 | 9–0–4 | 1.000 | 0.732 | [-0.000, +0.011] |
| AUPRC | N-LDF − SMOTE-r0.75 | -0.0090 | 5–1–7 | 1.000 | 0.868 | [-0.024, +0.002] |
| AUPRC | N-LDF − ROS-r0.75 | -0.0050 | 9–0–4 | 1.000 | 0.868 | [-0.021, +0.008] |

## Family `mechanism_parity`

| metric | contrast (a − b) | mean diff | W–T–L (tasks) | source Holm p | task Holm p | cluster 95% interval |
|---|---|---:|---:|---:|---:|---:|
| F1 | N-LDF-r1.0 − SMOTE | -0.0611 | 2–0–11 | 0.211 | 0.144 | [-0.122, -0.007] |
| F1 | SMOTE − ROS-r1.0 | +0.0204 | 8–0–5 | 0.193 | 0.220 | [+0.004, +0.044] |
| F1 | N-LDF-r1.0 − ROS-r1.0 | -0.0408 | 5–0–8 | 0.625 | 0.588 | [-0.107, +0.022] |
| AUPRC | SMOTE − ROS-r1.0 | +0.0048 | 9–0–4 | 0.480 | 0.382 | [-0.000, +0.012] |
| AUPRC | N-LDF-r1.0 − ROS-r1.0 | -0.0044 | 9–0–4 | 1.000 | 0.910 | [-0.023, +0.011] |
| AUPRC | N-LDF-r1.0 − SMOTE | -0.0092 | 6–1–6 | 1.000 | 0.910 | [-0.026, +0.003] |

## Family `threshold`

| metric | contrast (a − b) | mean diff | W–T–L (tasks) | source Holm p | task Holm p | cluster 95% interval |
|---|---|---:|---:|---:|---:|---:|
| F1 | Baseline-thr − Baseline | +0.0578 | 8–0–5 | 0.316 | 0.229 | [+0.017, +0.102] |
| F1 | SMOTE − Baseline-thr | +0.0383 | 10–0–3 | 0.258 | 0.229 | [+0.006, +0.081] |
| F1 | N-LDF-r1.0 − Baseline-thr | -0.0229 | 6–0–7 | 1.000 | 1.000 | [-0.068, +0.023] |
| F1 | N-LDF − Baseline-thr | -0.0238 | 7–0–6 | 1.000 | 1.000 | [-0.070, +0.021] |
| AUPRC | Baseline-thr − Baseline | -0.0002 | 0–9–4 | 1.000 | 0.500 | [-0.000, +0.000] |
| AUPRC | N-LDF − Baseline-thr | +0.0017 | 7–0–6 | 1.000 | 1.000 | [-0.007, +0.012] |
| AUPRC | N-LDF-r1.0 − Baseline-thr | +0.0003 | 6–0–7 | 1.000 | 1.000 | [-0.009, +0.009] |
| AUPRC | SMOTE − Baseline-thr | +0.0095 | 5–0–8 | 1.000 | 1.000 | [-0.009, +0.035] |
