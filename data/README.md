# data/

This directory is empty on purpose. The dataset is regenerated rather than shipped.

```bash
make data          # deterministic synthetic dataset, ~40s, no credentials
make data-kaggle   # …or the real Kaggle cats-vs-dogs archive
make preprocess    # 224x224 RGB, hashed 80/10/10 split, manifest + digests
```

`data/dataset.lock.json` is kept in the repository: it is the content-hash index of
the tree that produced the shipped model, so you can verify a regenerated dataset
matches the one behind `artifacts/metrics/baseline.json` before comparing numbers.

The generator is seeded, so `make data` reproduces the same images byte for byte.
