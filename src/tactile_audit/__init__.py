"""Public imports for the preserved tactile perturbation implementations.

This facade does not change the kernels, random streams, or assignment costs.
Training dependencies are optional and are not imported here.
"""

from icassp_jitter_operators import (
    active_cell_audit,
    cf_matched,
    exact_loss_matched_dropout,
    make_deconfounded_bundle,
    make_jitter_primitives,
    nested_cf,
    nested_pbj,
    prebin_jitter,
)
from injective_pbj_operator import make_injective_pbj_controls

__version__ = "0.2.0"

__all__ = [
    "active_cell_audit",
    "cf_matched",
    "exact_loss_matched_dropout",
    "make_deconfounded_bundle",
    "make_injective_pbj_controls",
    "make_jitter_primitives",
    "nested_cf",
    "nested_pbj",
    "prebin_jitter",
]
