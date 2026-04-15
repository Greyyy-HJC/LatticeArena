"""Measurement pipeline sketch for boosted pion two-point correlators.

Intended workflow:
1. Load gauge configuration and quark propagators.
2. Call operator.setup() once per configuration.
3. For each source time and target momentum (focus: |p| ~= 1 GeV), call
   operator.build() to obtain source/sink profiles and Dirac structure.
4. Form C_pi(p, t) and save correlators for benchmark metric extraction.
"""
