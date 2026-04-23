# Archived Wrong-Probe Code

This directory contains the retired probing / paper-analysis code that was used
before the `2026-04-23` methodology incident was identified.

Why this was archived:
- the `probe_physprobe.py` family used PEZ-style episode aggregation on
  manipulation episodes
- that made the main kinematic probe results episode-summary decoding rather
  than time-aligned dynamics decoding
- the current repo reset is focused on dataset recollection, value auditing,
  and dataset re-release

What remains active in the repo:
- dataset collection under `archive_data_collection/`
- shim packages `envs/`, `mdp/`, `policies/` used by the collector
- dataset / value-audit outputs needed for regeneration

What was moved here:
- old probe scripts
- cross-model / functional / scale-law analysis helpers
- obsolete experiment runner shells
- contaminated probe/result artifacts from the retired pipeline
