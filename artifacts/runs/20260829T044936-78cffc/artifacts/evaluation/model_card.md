# Model card — catsdogs-classifier

**Task.** Binary image classification, cat vs dog, for a pet adoption platform.

## Model
| Field | Value |
| --- | --- |
| Type | mlp |
| Input geometry | 224x224 RGB, downscaled to 24x24 |
| Features | 1728 |
| Classes | cat, dog (index order) |
| Decision threshold | 0.5 |
| Epochs run | 5 |
| Training rows | 958 |
| Seed | 42 |
| Trained at | 2026-08-29T04:49:35+00:00 |
| Dataset digest | `6fd7ebf8d3136103` |
| Git commit | `n/a` |

## Test-split metrics
| Metric | Value |
| --- | --- |
| accuracy | 0.5636 |
| precision | 0.5588 |
| recall | 0.6786 |
| f1 | 0.6129 |
| roc_auc | 0.6918 |
| log_loss | 3.6743 |

Evaluated on 55 held-out images.

## Confusion matrix
Rows are true classes, columns predicted: `[[12, 15], [9, 19]]`

## Environment
| Component | Version |
| --- | --- |
| python | 3.12.3 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| platform | Linux-6.18.44-fc-v22-x86_64-with-glibc2.39 |

## Intended use and limits
Intended as a first-pass triage aid for adoption listings, not an authority. It was
trained on a small, evenly balanced set and has seen no photographs of animals other
than the two classes; anything else is forced into one of them. Confidence is
calibrated only against the training distribution, so a low-confidence prediction
should route to a human rather than to a default.

_Generated 2026-08-29T04:49:36+00:00._
