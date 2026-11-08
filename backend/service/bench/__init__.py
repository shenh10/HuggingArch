"""Kernel Bench measurements in a DuckLake (docs/measured_calibration.md §4).

Three stored tables — `batches`, `inputs`, `measurements` — and nothing
derived: every view is computed per request. An input's annotations (its
grouping labels) are computed in memory at startup and for each new input
(`store.annotate_all`, the `annot` table), never stored. Modules:

  lake      the DuckDB instance attached to the lake (one process,
            root-only directories); reads, write transactions with retry,
            the owner / inputs / maintainer / backup locks, the free-space check
  store     writes: upload, review state, deletion; the in-memory annotations
  annotations the input rules and their revision (`rules_rev`) that decide
            which rows a lookup key selects
  browse    the Kernel Bench page's views (envelope, rows, workload, curves,
            coverage, facets)
  sets      the batch pages: list, summary, rows, overlap, nominations
  maintain  the paced loop: lake maintenance, backups
  backup    snapshots (content-addressed data, catalog copy, manifest,
            `current`) to the backup bucket; retention; restore
  serving   the startup gate: restore, open, lease, annotations, backups;
            whether the lake may be read and written now

  lookup    the measurements an estimate reads (rows per requested key, at a
            pinned snapshot), the owner's calibration report, the snapshot
            the shared rows last changed at
  routes    the HTTP routes (`/api/bench/*`, `/internal/bench/*`)
  identity  who a request is for: the headers the HA API forwards
  server    the Kernel Bench service: `python -m backend.service.bench.server`,
            the one process that opens the lake (docs/measured_calibration.md §5.1)

The store knows no users or permissions: the HTTP routes resolve who may see
or change what and pass explicit owners and scopes.
"""
