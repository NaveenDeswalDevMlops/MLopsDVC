# Model card — catsdogs-classifier

**Task.** Binary image classification, cat vs dog, for a pet adoption platform.

## Model
| Field | Value |
| --- | --- |
| Type | logreg |
| Input geometry | 224x224 RGB, downscaled to 24x24 |
| Features | 1728 |
| Classes | cat, dog (index order) |
| Decision threshold | 0.5 |
| Epochs run | 10 |
| Training rows | 958 |
| Seed | 42 |
| Trained at | 2026-08-29T05:09:16+00:00 |
| Dataset digest | `6fd7ebf8d3136103` |
| Git commit | `n/a` |

## Test-split metrics
| Metric | Value |
| --- | --- |
| accuracy | 0.7818 |
| precision | 0.7667 |
| recall | 0.8214 |
| f1 | 0.7931 |
| roc_auc | 0.8638 |
| log_loss | 0.5155 |

Evaluated on 55 held-out images.

## Confusion matrix
Rows are true classes, columns predicted: `[[20, 7], [5, 23]]`

## Environment
| Component | Version |
| --- | --- |
| python | 3.11.16 |
| numpy | 2.1.3 |
| scikit-learn | 1.5.2 |
| joblib | 1.4.2 |
| platform | macOS-26.6-arm64-arm-64bit |

## Intended use and limits
Intended as a first-pass triage aid for adoption listings, not an authority. It was
trained on a small, evenly balanced set and has seen no photographs of animals other
than the two classes; anything else is forced into one of them. Confidence is
calibrated only against the training distribution, so a low-confidence prediction
should route to a human rather than to a default.

_Generated 2026-08-29T05:09:19+00:00._
