# Completed training pipeline

The scaffold has been replaced by a working, executed training pipeline.

See [the project README](../README.md) for the architecture, reproduction commands,
measured results, and limits. [results.json](results.json) contains the full report.
The current browser app loads the trained artifacts from `dist/model/` and performs
real inference through the published 499-neuron MaleCNS subgraph.
