"""Portable adapters over the preserved historical operator implementations."""
import numpy as np

import icassp_stage_b_runner as stage_b
import run_icassp_jitter_screening as screen
from injective_pbj_operator import make_injective_pbj_controls
from retention_matched_operator import make_retention_matched_controls
from survivor_rule_operator import ALTERNATIVE_CONDITIONS, make_survivor_rule_controls

TRAIN_ARMS = {"clean", "pbj_aug", "cf_aug", "prebin_aug", "matched_aug", "cf_matched_aug"}
BASE_CONDITIONS = {"clean", "prebin", "pbj", "cf", "matched", "cf_matched"}
INJECTIVE_CONDITIONS = {"ipbj_movement", "pbj_identity_loss", "ipbj_plus_identity_loss"}
RETENTION_CONDITIONS = {f"retention_{percent}" for percent in (95, 90, 85)}
CONDITIONS = BASE_CONDITIONS | INJECTIVE_CONDITIONS | set(ALTERNATIVE_CONDITIONS) | RETENTION_CONDITIONS


class TrainingDataset(stage_b.StageBTrainingDataset):
    """Add the two historical matched-augmentation arms without monkey-patching."""

    def __getitem__(self, item):
        if self.job.train_arm not in {"matched_aug", "cf_matched_aug"}:
            return super().__getitem__(item)
        index = int(self.indices[item])
        clean = self.screen.unpack_clean_sample(self.bundle, index)
        augmented, severity, operator_seed = stage_b.augmentation_decision(
            self.registry, self.job, self.epoch, index
        )
        output = clean
        if augmented:
            arrays, _ = self.adapter.postbin_bundle(clean, severity // 25, operator_seed, 0)
            output = arrays[{"matched_aug": "matched", "cf_matched_aug": "cf_matched"}[self.job.train_arm]]
        return np.asarray(output, dtype=np.float32), int(self.bundle.labels[index]), index


def operator_seed(bundle, index, registry, *, retention=False):
    if bundle.name == "braille" or retention:
        return stage_b.stable_seed(registry["protocol_id"], "operator_root_v1", bundle.name, bundle.sample_ids[index])
    return screen.stable_seed(screen.PROTOCOL_ID, "operator_root_v1", bundle.name, bundle.sample_ids[index])


def generate_condition(registry, adapter, bundle, index, condition, severity, realization):
    if condition in BASE_CONDITIONS:
        if bundle.name == "braille":
            return stage_b.generate_stage_b_condition_sample(registry, screen, adapter, bundle, index, condition, severity, realization)
        return screen.generate_condition_sample(adapter, bundle, index, screen.ConditionSpec(condition, severity, realization), 0)
    clean = screen.unpack_clean_sample(bundle, index)
    if condition in RETENTION_CONDITIONS:
        percent = int(condition.split("_")[1])
        generated = make_retention_matched_controls(
            clean, dataset=bundle.name,
            sample_operator_seed=operator_seed(bundle, index, registry, retention=True),
            realization=realization, retention_percentages=(percent,),
        )
        return generated["outputs"][percent], generated["audits"][percent]
    seed = operator_seed(bundle, index, registry)
    if condition in INJECTIVE_CONDITIONS:
        generated = make_injective_pbj_controls(clean, severity // 25, seed, realization=realization)
    elif condition in ALTERNATIVE_CONDITIONS:
        generated = make_survivor_rule_controls(clean, severity // 25, seed, realization=realization)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return generated[condition], generated["audit"]
