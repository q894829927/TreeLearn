"""Main dataset with independently seeded augmentation and a fixed sample plan."""
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from tree_learn.dataset import TreeDataset
from .provenance import sha256, write_json, canonical_hash


def create_plan(crop_manifest, output, seed, epochs=1000, examples=500):
    import json
    manifest = json.loads(Path(crop_manifest).read_text())
    paths = manifest["crops"]
    if manifest.get("status") != "verified" or len(paths) != 25000:
        raise ValueError("A verified 25,000-crop manifest is required for a frozen plan")
    rng = np.random.RandomState(seed)
    indices, augmentation = [], []
    for _ in range(epochs):
        indices.append(rng.permutation(len(paths))[:examples])
        augmentation.append(rng.randint(0, 2**32 - 1, size=examples, dtype=np.uint32))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, indices=np.array(indices), augmentation=np.array(augmentation))
    write_json(str(output) + ".json", {"seed": seed, "epochs": epochs, "examples_per_epoch": examples,
               "crop_manifest_sha256": sha256(crop_manifest), "plan_sha256": sha256(output),
               "paths_hash": canonical_hash([item["path"] for item in paths])})


def resource_indices(sample, maximum, seed):
    """Shared resource variant: full supervision first, then a common index set.

    Preserve XY extremes and up to 16 bottom-band points per visible tree,
    reserve additional instance points, then fill uniformly without replacement.
    The sample's spatial extent and small instances are prioritized. A cap can
    still discard instances when the number of instances exceeds the budget.
    """
    coords, _, labels = sample[:3]
    n = len(coords)
    if maximum is None or n <= maximum:
        return np.arange(n)
    rng = np.random.RandomState(seed ^ 0x70545633)
    xyz, inst = coords.numpy(), labels.numpy()
    mandatory = set(np.argmin(xyz[:, :2], axis=0).tolist() + np.argmax(xyz[:, :2], axis=0).tolist())
    for label in np.unique(inst[inst > 0]):
        members = np.flatnonzero(inst == label)
        bottom = members[xyz[members, 2] <= xyz[members, 2].min() + 0.5]
        mandatory.update(rng.choice(bottom, min(16, len(bottom)), replace=False).tolist())
        mandatory.update(rng.choice(members, min(16, len(members)), replace=False).tolist())
    keep = np.array(sorted(mandatory), dtype=np.int64)
    if len(keep) > maximum:
        raise ValueError("Point cap too small to retain required spatial/instance samples")
    available = np.setdiff1d(np.arange(n), keep, assume_unique=True)
    return np.sort(np.concatenate([keep, rng.choice(available, maximum - len(keep), replace=False)]))


class AuditedSample(tuple):
    """Nine main tensors plus non-tensor provenance carried through collation."""


class PlannedDataset(TreeDataset):
    def __init__(self, paths, inner_square_edge_length=8, training=True,
                 data_augmentations=None, maximum_points=None):
        # Do not call os.listdir: paths follow the audited manifest ordering.
        self.data_paths = list(map(str, paths))
        self.inner_square_edge_length = inner_square_edge_length
        self.training = training
        self.data_augmentations = data_augmentations
        self.logger = logging.getLogger("ptv3_treelearn")
        self.maximum_points = maximum_points

    def __getitem__(self, key):
        index, augmentation_seed = key if isinstance(key, tuple) else (key, 0)
        state = np.random.get_state()
        try:
            np.random.seed(augmentation_seed)
            sample = super().__getitem__(index)
        finally:
            np.random.set_state(state)
        indices = resource_indices(sample, self.maximum_points, augmentation_seed)
        result = AuditedSample(tensor[indices] for tensor in sample)
        before_labels, after_labels = sample[2].numpy(), result[2].numpy()
        before = np.unique(before_labels[before_labels > 0])
        after = np.unique(after_labels[after_labels > 0])
        result.audit = {"path": self.data_paths[index], "augmentation_seed": augmentation_seed,
                        "points_before": len(sample[0]), "points_after": len(indices),
                        "instances_before": len(before), "instances_after": len(after),
                        "instance_retention": len(after)/len(before) if len(before) else None,
                        "xyz_extent_before": torch.max(sample[0], 0).values.sub(torch.min(sample[0], 0).values).tolist(),
                        "xyz_extent_after": torch.max(result[0], 0).values.sub(torch.min(result[0], 0).values).tolist(),
                        "supervision_order": "main augmentation -> main getOffset/masks on full crop -> shared sample indices"}
        return result

    def collate_fn(self, batch):
        result = super().collate_fn(batch)
        result["sampling_audit"] = [sample.audit for sample in batch]
        return result


class PlanSampler(Sampler):
    def __init__(self, indices, augmentation, start=0):
        self.indices = indices.reshape(-1)
        self.augmentation = augmentation.reshape(-1)
        self.start = start

    def __iter__(self):
        for i in range(self.start, len(self.indices)):
            yield int(self.indices[i]), int(self.augmentation[i])

    def __len__(self):
        return len(self.indices) - self.start


def sample_digest(batch):
    import hashlib
    digest = hashlib.sha256()
    for name in ("coords", "input_feats", "instance_labels", "semantic_labels", "offset_labels", "masks_sem", "masks_off"):
        digest.update(batch[name].contiguous().numpy().tobytes())
    return digest.hexdigest()
