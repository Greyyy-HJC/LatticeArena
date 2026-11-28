# Exact source snapshot for the D-development replay

This directory preserves the 15-file local Python import closure used by the numerical replay. Paths retain their repository layout so the snapshot can be placed on `PYTHONPATH` without importing the active task modules from a different revision. `source_manifest.json` records the source revision and SHA-256 for every copied module. The semantic protocol input reconstructed from D provenance is archived one level up as [`replay_protocol.json`](../replay_protocol.json); its canonical measurement fingerprint is recorded in the replay audit.

The snapshot is code provenance, not a standalone runnable package. It does not include `development.npz`, `development_provenance.json`, per-bootstrap arrays, or a pinned Python/NumPy/SciPy/Matplotlib environment. The open D input bundle must be supplied separately; no E anchor, gap, validation, confirmation, or shared-holdout data is included.
