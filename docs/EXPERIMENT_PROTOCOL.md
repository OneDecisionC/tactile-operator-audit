# Experiment protocol

This document describes the historical experiments reported in *Occupancy-Matched Auditing of Temporal Jitter in Binary-Binned Tactile Event Representations*, revision `polished_revision_20260908_v3`. Version 0.2.0 adds software validation and tiny synthetic computations, not new paper measurements. See VALIDATION.md for the actual checks.

## Data and splits

Braille uses the released `data_braille_letters_th1` event-encoding version: 5,400 balanced samples, 27 classes, and 12 taxels represented as 24 ON/OFF channels. The inclusive observation window is [0.175, 1.300] seconds, rebased to [0, 1.125] seconds. The recognition input has 45 time bins of 25 ms. Events exactly at the right endpoint enter the final bin. The dataset files used by this study do not expose participant/session identifiers, so the protocol uses sample-level class-stratified five-fold splits. This does not establish session-independent generalization.

STEMNIST uses released `ProcessedSpikes`: 7,700 samples, 35 classes, 34 participants, and 256 taxels represented as 512 ON/OFF channels. The inclusive window is [0, 2.000] seconds. Recover integer 120-Hz ticks before binning into 80 time bins of 25 ms. Splits are participant-disjoint five-fold splits.

Both datasets are offline eventizations of frame-based tactile recordings. Split seeds are 42, 123, and 202. Original sample membership, checkpoint files, and upstream random-key manifests are not included in this candidate. A newly generated split is not a historical split merely because it uses the same integer seed.

Data must be obtained separately from the source records: [Braille, version 1.1](https://doi.org/10.5281/zenodo.7050094) and [STEMNIST, version v1](https://doi.org/10.5281/zenodo.19469535). This candidate does not redistribute those datasets.

## Checkpoints and training lineage

| Dataset | Paper probe | Parameters | Historical lineage |
|---|---|---:|---|
| Braille | RateSNN-h192-l3 | 47,067 | Stage-B formal |
| Braille | TCN-h64-b2-k3 | 45,595 | Stage-B formal |
| Braille | Conv1D-h83-l2-k5 | 47,171 | Stage-B formal |
| STEMNIST | paper-topology SCNN | 6,507 | Stage-A frozen screen |
| STEMNIST | TCN-h128-l3 | 514,851 | Stage-A frozen screen |

Braille uses AdamW, learning rate 0.002, weight decay 0.0001, global-norm clipping at 1.0, 80 epochs, and batch size 128. STEMNIST uses Adam, the same learning rate and weight decay, no gradient clipping, 50 epochs, and batch size 32. Both use unweighted cross-entropy, no learning-rate scheduler, no early stopping, and the final checkpoint.

The STEMNIST TCN configuration records hidden width 128, three layers, and dropout 0.3. The SCNN records beta 0.9, threshold 1, dropout 0.3, and spike readout. "Paper-topology" describes network shape only: the 80-bin representation and participant-disjoint evaluation do not reproduce the original STEMNIST paper's protocol.

A checkpoint is fixed across test operators. The six augmentation training arms have separate weights. The clean-trained comparisons do not imply that all augmentation arms share one checkpoint.

## Representation and perturbations

Binary binning records whether at least one retained event falls in each time-channel cell. Raw-event count M and occupied-cell count N are different quantities; N <= M.

Recognition uses nominal jitter bounds J = 25, 50, and 75 ms, corresponding to radii R = 1, 2, and 3 bins. There are three replayable realizations. The pooled corruption accuracy averages all nine severity-realization cells.

PBJ draws a sign uniformly from {-1,+1} and a magnitude q uniformly from [0,1), then uses `sign * floor(q*R + 0.5)`. It clips proposals to the time window and writes them by logical OR. This is not discrete-uniform sampling over the 2R+1 integer shifts. Primitives are shared across severity within a sample and realization and depend on input shape.

CF performs one, two, or three adjacent-swap passes. Each pass draws a shared phase in {0,1}; disjoint legal adjacent pairs are swapped independently for each channel with probability 0.5. Different severities replay prefixes of the same plan. CF preserves clean count within each sample/channel, but does not match PBJ's movement distribution.

Matched deletion M removes exactly the PBJ occupancy-loss budget in each sample/channel. A frozen PCG64 rank chooses the deleted identities, with source time breaking ties. CF+M applies that deletion before the shared CF permutation. PBJ, M, and CF+M have identical retained counts for each sample/channel. Their movement kernels, deletion positions, boundary behavior, and local correlations can differ.

Pre-bin jitter perturbs retained timestamps uniformly within legal boundary-truncated intervals before rebinning. It is a different input operation from PBJ.

## Proposal-coupled and survivor controls

iPBJ reuses PBJ proposals and assigns each source a distinct legal destination within the original radius. Collision-free channels keep their proposals. Colliding channels use the frozen weighted linear-assignment objective recorded in `configs/paper_protocol.json`. The recorded solver version is SciPy 1.13.1.

The four survivor rules are `assignment_preserving`, `fixed_index`, `minimum_proposed_displacement`, and `fixed_random_priority`. They share the injective assignment and the per-sample/channel loss budget. The primary rule preferentially retains a source whose assignment matches its PBJ proposal.

OR does not identify a physically surviving source. Movement-only, source-deletion, and combined conditions are operational controls, not unique causal contributions. The primary rule is aligned with the assignment construction; a small PBJ-minus-combined residual is not independent validation or an equivalence result.

Retention calibration separately targets 95%, 90%, and 85% of the clean occupied cells in each sample. It retains `floor(rho*N/100 + 0.5)` cells, allocates deletions with a capped deterministic D'Hondt/Jefferson prefix, and uses target-independent SHA-256 priorities for nested within-channel deletions. This matches retention, not PBJ's deletion locality.

## Augmentation transfer

The Braille TCN has six separately trained arms: clean, PBJ, CF, pre-bin, M, and CF+M. An augmented sample is corrupted with probability 0.75; severity is sampled uniformly. The M and CF+M follow-ups reuse the training-arm-independent random-key schedule for initialization, epoch order, augmentation gate, severity, and operator root.

The archived transfer matrix includes all six train arms and six test conditions. A gain under one operator does not establish transfer to another. Training-arm accuracy differences are not additive causal shares.

## Accuracy and uncertainty

Each split seed concatenates five held-out folds into out-of-fold predictions. Main tables report the mean and sample standard deviation across the three split seeds. The standard deviation of a paired difference is computed historically from paired seed-level differences, not by subtracting the two standard deviations.

For fractional accuracies, interaction is `100*(A_combined - A_move - A_deletion + A_clean)`; residual is `100*(A_PBJ - A_combined)`. Their units are percentage points.

The historical paired bootstrap averages correctness within a sample across split seeds and corruption cells, then uses 10,000 resamples. Braille resamples samples within classes; STEMNIST resamples the 34 participant clusters. Intervals are pointwise and conditional on frozen OOF models and observed corruptions. They exclude refitting and overlapping-training-set uncertainty.

Table 1 uses sample-micro intervals, including participant-clustered sample-micro intervals for STEMNIST. The specified STEMNIST sensitivity analyses use participant-macro summaries. A confidence interval containing zero establishes neither equivalence nor the absence of an effect.

## Bin-width audit

The operator-only audit fixes J = 50 ms and the observation window while varying bin width over 12.5, 25, and 50 ms. The final Braille bin on the 50-ms grid has width 25 ms. Shape-dependent random draws are not paired eventwise across grids. Only the 25-ms input reproduces the historical recognition grid.

Clean occupancy ratio is the sum of occupied cells divided by the sum of retained raw events. PBJ retention pools occupied counts across samples and three realizations, divided by three times the clean occupied count. These are ratios of totals, not means of per-sample ratios. Error bars span three realization values and are not confidence intervals.

Reported displacement in milliseconds is bin displacement times the nominal bin width. Fixing J does not fix the quantized shift distribution across grids. The audit contains no cross-grid model evaluation.
