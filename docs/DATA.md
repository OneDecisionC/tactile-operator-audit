# Data acquisition and representation

Dataset binaries are deliberately excluded. Obtain the specified version from the original publisher, retain its citation and license, and keep the data outside version control.

## Braille

- Dataset: Tactile Braille Letters Dataset, version 1.1.
- Version DOI: [10.5281/zenodo.7050094](https://zenodo.org/records/7050094).
- Related paper: [Muller-Cleve et al., Frontiers in Neuroscience, 2022](https://doi.org/10.3389/fnins.2022.951164).
- The manuscript uses the released `th1` event-encoding variant, not a later dataset version with a different acquisition rate.
- The historical loader expects `<data-project-root>/braille_letters_dataset/data/data_braille_letters_th1`. Follow the local path settings documented in `RUNNING_EXPERIMENTS.md`; do not silently substitute a different event threshold.
- Paper representation: 27 classes, 5,400 samples, 12 taxels unfolded into 24 ON/OFF channels, 45 bins of 25 ms after the stated crop.
- The frame-based recordings are eventized offline. Do not call them native asynchronous sensor timestamps without that qualification.

## STEMNIST

- Dataset: STEMNIST: Spiking Tactile Extended MNIST Neuromorphic Dataset, version v1.
- Version DOI: [10.5281/zenodo.19469535](https://zenodo.org/records/19469535).
- Related paper: [Tripathi et al., arXiv:2601.01658](https://arxiv.org/abs/2601.01658).
- Use the released `ProcessedSpikes` input expected by the loader. Data-cache construction uses the original loading and binning code; it is real data processing, not a harmless documentation command.
- Paper representation: 7,700 samples, 35 classes, 34 participants, 256 taxels separated into 512 polarity channels, 80 bins of 25 ms.
- Participant-disjoint folds and the 512-channel representation are choices in the present study, not claims that every original-paper setting was reproduced.

## Before computing

1. Record the exact version DOI and the downloaded archive's checksum.
2. Confirm the extracted layout and event-encoding variant using the original dataset documentation.
3. Set local data/cache/output paths in your own configuration. Do not commit private absolute paths.
4. Preserve the frozen crop, timestamp boundary handling, bin width, channel order, and split rules.
5. Keep a manifest identifying the inputs used by each new run. No new archive checksum or data validation was performed during release preparation.

## Rights, privacy, and trusted inputs

The official records list CC BY 4.0. This is a dataset reuse license, not an ethics approval or a license for every code file in this repository. Verify the original records and applicable institutional requirements before redistribution.

Participant-level predictions, original sample identities, dataset archives, and caches are excluded from this package. The shared CSVs are curated aggregates. A future decision to release more granular artifacts requires a separate privacy and rights review.

Some historical loaders/checkpoints use pickle-compatible serialization. Only load data and checkpoints from sources you trust. Do not treat an arbitrary downloaded `.pkl` or `.pt` file as a harmless text document.
