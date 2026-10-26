# GapSBI Release Documentation

This directory records the public-facing workflow for the validated GapSBI
benchmark. It describes existing implementations and artifacts; it does not
replace their provenance metadata or alter historical results.

* [Canonical workflows and scope](workflows.md): benchmark scope, authoritative
  entry points, evaluation targets, naming conventions, diagnostics, and
  migration boundaries.
* [Benchmark and reproduction guide](benchmark.md): posterior targets,
  task-specific references, commands, reliability reporting, and limitations.
* [Reference artifact documentation](../references/README.md): HDF5 contracts,
  task-specific full-reference procedures, and reference diagnostics.
* [Partial-reference artifact documentation](../references/partial_data_references_v1/README.md):
  stored partial-data conditioning variables and sparse LV/MNAR provenance.

Reference metadata, committed configurations, and the pinned Conda environment
provide the reproducibility boundary for the validated release.
