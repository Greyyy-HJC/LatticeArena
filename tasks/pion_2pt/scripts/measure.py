"""Measurement pipeline for boosted pion two-point correlators (not yet implemented).

Once implemented, this script will provide the fixed physics pipeline:

1. Load gauge configuration from ``dataset/``
2. Call ``submission.setup(gauge_field, latt_size, lattice_spacing_fm)``
3. For each target momentum and source timeslice:
   a. Call ``submission.build(gauge_field, momentum_gev, t_source)`` to obtain
      source/sink profiles and Dirac structure
   b. Construct the quark source using the submission's spatial profile
   c. Invert the Dirac operator to obtain quark propagators (PyQUDA)
   d. Contract propagators to form the pion two-point correlator
4. Average over source times and return ``C_pi(p, t)``

The benchmark will call this pipeline and extract metrics from the resulting
correlators (SNR, effective mass plateau quality, excited-state contamination).
"""
