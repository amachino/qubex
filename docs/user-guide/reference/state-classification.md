# State classification

Qubex supports readout classification using k-means or Gaussian mixture models (GMM). Classifiers are stored per qubit and reused during measurement post-processing.

## Classifier types

- **k-means**: Simple clustering for state separation.
- **GMM**: Gaussian mixture model with probabilistic outputs.

Select the classifier type with `classifier_type` when creating an `Experiment`.
(`Measurement` also supports this for advanced usage.)

## Storage layout

Classifiers are loaded from the classifier directory (default: `.classifier`) using the pattern:

```
.classifier/<chip_id>/<qubit>.pkl
```

Qubex automatically loads available classifiers into `Experiment.classifiers` and uses them for single-shot classification.

## Using classification in results

```python
result = exp.measure(sequence, mode="single", shots=4096)

measure_data = result.data[exp.qubit_labels[0]]
probs = measure_data.probabilities
counts = measure_data.counts
```

Classification utilities require `mode="single"` and a loaded classifier for the target.

## Export classified data

```python
result.save_classified(
    "classified.json",
    targets=["Q00", "Q01"],
    threshold=0.8,
    format="json",
)
```
