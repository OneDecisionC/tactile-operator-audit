# Operator API and scientific semantics

These modules are a faithful snapshot of the study implementation. The scientific
kernels are copied without refactoring, renaming, or changing random streams.

## Source and version

- `src/icassp_jitter_operators.py`: copied from `icassp_jitter_operators.py` in the
  original `Project/tactileDataset` project root.
- `src/injective_pbj_operator.py`: copied from
  `injective_pbj_v1/injective_pbj_operator.py` in the study directory dated
  2026-08-19. Its actual `OPERATOR_VERSION` is `injective_pbj_controls_v2`.
  The old directory suffix `v1` does not supersede that code version.
- The pre-bin audit identifies
  `prebin_timestamp_jitter_conditional_uniform_v2`; the other audit version
  strings remain exactly as recorded by the original implementation.
- The release is covered by the author-confirmed [MIT License](../LICENSE).
  Per-file headers have not been added, preserving the scientific kernel bytes
  and their recorded provenance hashes.

Only NumPy, SciPy, and the Python standard library are needed by these two
modules. Python 3.10 or later is required by the annotations.

## Post-bin inputs and units

Each call accepts one NumPy array `clean` of shape `[T, C]`, with positive
dimensions, dtype `numpy.uint8` or `numpy.float32`, and values exactly 0 or 1.
Post-bin outputs retain the input dtype. There is no batch dimension.
`T` is the number of time bins; `C` is the number of channels after any
caller-defined spatial and polarity mapping.

The post-bin argument `radius_bins` is a nonnegative integer, not milliseconds.
For bin width `delta_ms`, an exactly representable physical radius satisfies
`radius_bins = J_ms / delta_ms`. For example, 10 ms bins and 50 ms nominal
severity give `radius_bins = 5`. If this ratio is not an integer, use the
conversion already fixed by the study caller; these kernels do not define a
new millisecond-to-bin rounding rule. The PBJ kernel separately rounds its
scaled random displacement half away from zero.

The pre-bin interface uses seconds for both timestamps and severity.
For example, 50 ms is `severity_seconds=0.050`. Bin width is
`duration_seconds / n_steps`. Pre-bin input records can contain multiple
events that later occupy the same binary cell.

## Public interfaces

```python
nested_pbj(clean, radius_bins, seed, *, realization=0, return_audit=False)
nested_cf(clean, radius_bins, seed, *, realization=0, return_audit=False)
exact_loss_matched_dropout(clean, pbj, seed, *, realization=0, return_audit=False)
cf_matched(clean, pbj, radius_bins, seed, *, realization=0, return_audit=False)

make_deconfounded_bundle(clean, radius_bins, seed, *, realization=0)
make_jitter_primitives(shape, max_cf_passes, seed, *, realization=0)
active_cell_audit(source, corrupted)

prebin_jitter(
    timestamps, channels, severity_seconds, duration_seconds, seed,
    *, n_steps, n_channels=None, realization=0, return_timestamps=False,
)

make_injective_pbj_controls(clean, radius_bins, seed, *, realization=0)
```

The first four functions return the corrupted array, or `(array, audit)`
when `return_audit=True`. `make_deconfounded_bundle` returns the keys
`pbj`, `cf`, `matched`, `cf_matched`, and `audit`.

`prebin_jitter` returns `(occupancy, audit)` by default or
`(occupancy, jittered_timestamps, audit)` with `return_timestamps=True`.
Its occupancy has dtype `uint8` and shape `[n_steps, n_channels]`.
Pass `n_channels` explicitly for an empty event stream and whenever silent
channels must be retained. Channels are nonnegative integer indices;
timestamps must be finite and within `[0, duration_seconds]`.

`make_injective_pbj_controls` returns `ipbj_movement`, `pbj_identity_loss`,
`ipbj_plus_identity_loss`, `pbj`, and `audit`.

The core module also exports these aliases:

- `postbin_jitter = nested_pbj`
- `collision_free_jitter = nested_cf`
- `loss_matched_dropout = exact_loss_matched_dropout`
- `prebin_timestamp_jitter = prebin_jitter`

## Random coupling and replay

The core derives a stable seed using BLAKE2b from
`(seed, realization, operator namespace, shape)` and uses NumPy PCG64.
Severity is deliberately excluded. PBJ severities reuse signed normalized
displacements, matched-dropout conditions reuse fixed random ranks, and CF
severities reuse prefixes of one swap plan.

The kernels do not receive a sample identifier. Calls with equal shape, seed,
and realization reuse the same random primitives even for different samples.
The caller must preserve the study's per-sample seed and realization policy.
Do not introduce a different hash, reorder random draws, replace PCG64, or
include severity in the seed while claiming replay of the original study.

Shared primitives do not imply monotone accuracy, monotone PBJ collision
loss, or equal effective displacement distributions across operators.
Record the NumPy and SciPy versions along with audit digests when preparing
reproducible results; a solver tie can depend on the numerical implementation.

## Operator meanings

### PBJ

`nested_pbj` proposes an integer displacement for each active cell, clips
the target to the legal time range, and writes binary 1 at that location.
Multiple proposals to one target in one channel merge by logical OR.
The output therefore combines movement, clipping, and possible occupancy loss.
Clipping itself retains a target; it does not independently delete a record.

### CF

`nested_cf` applies `radius_bins` adjacent-swap passes. Each pass draws one
phase, 0 or 1, to form disjoint neighboring time-bin pairs, and draws a
Bernoulli-1/2 exchange gate for each pair and channel. The swap plan is
independent of the input values. A gate can select a pair whose equal values
produce no visible occupancy change.

Every pass is a channel-wise permutation, so CF exactly preserves each
channel's active-cell count. A source can move by at most one bin per pass.
`selected_swaps_by_pass` counts selected gates; it is not the number of
active cells that actually moved. Equal nominal radii for CF and PBJ do not
calibrate their realized movement distributions.

### Matched dropout and CF+M

`exact_loss_matched_dropout` takes both the clean array and its paired PBJ
array. In every channel, it deletes exactly the clean-minus-PBJ count using
fixed random ranks, with source time as the deterministic tie-break.

`cf_matched` performs matched deletion first and then CF, using the same
CF prefix plan. Both matched controls have exactly the PBJ output count
separately for every sample and channel. This matches occupancy counts,
not the PBJ source identities or temporal geometry.

### Pre-bin jitter

`prebin_jitter` canonicalizes records by timestamp and channel before assigning
shared uniform quantiles. At each severity, an event's quantile is mapped
into its legal displacement interval. This is conditional-uniform sampling
inside the recording window, not clipping an out-of-window uniform proposal
onto an endpoint. Exact duplicate records are treated as an indistinguishable
multiset. Returned jittered timestamps are in caller record order.

The raw number of events is preserved. Subsequent binary binning can merge
records, so raw-event conservation does not imply binary-occupancy
conservation. An event at the right endpoint belongs to the last bin.

### Proposal-coupled injective controls

`make_injective_pbj_controls` reuses the exact PBJ proposals. Collision-free
channels take those proposals directly. For collision channels, SciPy's
`linear_sum_assignment` assigns distinct legal destinations using the frozen
weighted cost and tie-break terms. The movement-only result conserves
every channel's clean count and retains the original displacement bound.

The source code's integer weights and tie-break rules must remain unchanged.
Describe this as the implemented weighted assignment; the public API does
not assert a general lexicographic-optimality theorem for arbitrary shapes.
The weight comments in the source assume the study regime `T, N <= 80`,
where `N` is the number of active sources in a channel. Extending the solver
to a different regime requires a separate scientific assessment.

Binary OR does not reveal a unique surviving source identity. The module
uses a fixed operational survivor convention based on repaired proposals,
displacements, and source time. `pbj_identity_loss` removes the resulting
lost identities at their original locations;
`ipbj_plus_identity_loss` removes those identities after injective movement.

The two identity-loss controls match PBJ's per-channel counts, but their
arrays need not equal PBJ. The audit explicitly reports uncovered PBJ
targets, repair distance, exact proposal assignments, and the combined
control's Hamming difference from PBJ. Do not describe this construction
as an exact recovery of physical event deaths or an additive decomposition
of accuracy loss.

## Demonstration

`examples/operator_demo.py` builds a small artificial binary array, calls
the bundle and injective interfaces, and prints the actual channel counts
and selected audit values. It contains no assertions and does not load
datasets or train a model. It was executed during release checks; the separate test suite adds assertions and historical synthetic regression checks.

After installing this repository, the demonstration can be invoked with:

```console
python examples/operator_demo.py
```

Existing defensive checks inside the preserved kernels are part of the
implementation. Copying those checks is not a claim that release tests
or behavioral validation have been run.
