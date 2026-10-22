"""Audits and independent crop generation using main's unchanged algorithms."""
import json
import logging
from pathlib import Path

import laspy
import numpy as np

from tree_learn.util.data_preparation import load_data
from tree_learn.util.parser import get_config
from . import MAIN_COMMIT, PTV3_COMMIT
from .provenance import sha256, write_json, environment, git_state, seed_all

FORESTS = "A1N A1W G1N G1W G2N G2W G3N G3W G4N G4W L1N L2N L2W LG1 LG2 LG3 O1N O1W".split()


def label_statistics(points, labels):
    values, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    minima = np.full(len(values), np.inf)
    maxima = np.full(len(values), -np.inf)
    np.minimum.at(minima, inverse, points[:, 2])
    np.maximum.at(maxima, inverse, points[:, 2])
    extents = (maxima - minima)[values > 0]
    return {"points": len(labels), "instances": int((values > 0).sum()),
            "non_tree_fraction": float(np.mean(labels == 0)), "ignore_fraction": float(np.mean(labels == -1)),
            "instance_point_count_quantiles": np.quantile(counts[values > 0], [0, .25, .5, .75, 1]).tolist() if np.any(values > 0) else [],
            "z_extent_quantiles_m": np.quantile(extents, [0, .25, .5, .75, 1]).tolist() if len(extents) else [],
            "bounds": [points.min(0).tolist(), points.max(0).tolist()]}


def audit_sources(source_root, output):
    root, output = Path(source_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"main_commit": MAIN_COMMIT, "ptv3_commit": PTV3_COMMIT,
                "source_root": str(root.resolve()), "git": git_state(), "environment": environment(),
                "training_forests": [], "evaluation": [], "weights": [],
                "existing_crops": {"count": len(list((root / "train/random_crops/npz").glob("*.npz"))),
                    "generation_seed": None, "accepted_for_training": False,
                    "reason": "Historical generation seed is not recorded; regenerate independently."}}
    for name in FORESTS:
        path = root / "train/forests" / (name + ".laz")
        row = {"forest": name, "path": str(path.resolve()), "sha256": sha256(path)}
        # Header/label audit avoids holding the dense raw forest in RAM.
        with laspy.open(path) as stream:
            row.update(points=stream.header.point_count, bounds=[stream.header.mins.tolist(), stream.header.maxs.tolist()],
                       fields=list(stream.header.point_format.dimension_names))
            counts, zmin, zmax = {}, {}, {}
            unknown = non_tree = 0
            for chunk in stream.chunk_iterator(2_000_000):
                tree = np.asarray(chunk.treeID)
                classes = np.asarray(chunk.classification)
                non = np.isin(classes, [1, 2])
                labels = np.where(non, 0, np.where(tree != 0, tree, -1))
                vals, ns = np.unique(labels, return_counts=True)
                unknown += int((labels == -1).sum())
                non_tree += int((labels == 0).sum())
                z = np.asarray(chunk.z)
                for label, count in zip(vals[vals > 0], ns[vals > 0]):
                    mask = labels == label
                    counts[int(label)] = counts.get(int(label), 0) + int(count)
                    zmin[int(label)] = min(zmin.get(int(label), float("inf")), float(z[mask].min()))
                    zmax[int(label)] = max(zmax.get(int(label), -float("inf")), float(z[mask].max()))
            row.update(instances=len(counts), non_tree_fraction=non_tree / row["points"], ignore_fraction=unknown / row["points"],
                       instance_point_count_quantiles=np.quantile(list(counts.values()), [0, .25, .5, .75, 1]).tolist(),
                       z_extent_quantiles_m=np.quantile([zmax[k]-zmin[k] for k in counts], [0, .25, .5, .75, 1]).tolist())
        manifest["training_forests"].append(row)
        write_json(output / "source_manifest.partial.json", manifest)
        print("audited", name, row["points"], row["instances"], flush=True)
    for name, rel in (("L1W_input", "pipeline/L1W/forest/L1W.laz"),
                      ("L1W_GT", "benchmark/L1W_voxelized01_for_eval.laz"),
                      ("Wytham_input", "pipeline/wytham/forest/wytham_vox0.1.laz"),
                      ("Wytham_GT", "benchmark/wytham_vox0.1.laz")):
        path = root / rel
        data = load_data(str(path))
        with laspy.open(path) as stream:
            fields = list(stream.header.point_format.dimension_names)
        row = {"name": name, "path": str(path.resolve()), "sha256": sha256(path), "fields": fields,
               **label_statistics(data[:, :3], data[:, 3])}
        manifest["evaluation"].append(row)
        write_json(output / "source_manifest.partial.json", manifest)
        print("audited", name, row["points"], row["instances"], flush=True)
    for name in ("hais_ckpt_spconv2.pth", "model_weights_20241213.pth", "model_weights_with_small_20241213.pth"):
        path = root / "model_weights" / name
        manifest["weights"].append({"name": name, "path": str(path.resolve()), "exists": path.exists(),
                                    "sha256": sha256(path) if path.exists() else None})
    write_json(output / "source_manifest.json", manifest)
    return manifest


def generate_crops(source_root, destination, seed=42, resume=False, workers=1):
    """Run main generation in a new directory; preserve all original labels.

    Directory iteration is sorted for a reproducible forest allocation order.
    Main's generate_random_crops algorithms are unchanged. The temporary wrapper
    only fixes os.listdir ordering under this destination and records the seed.
    """
    import os
    from tools.data_gen.gen_train_data import generate_random_crops
    destination = Path(destination)
    if destination.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True, exist_ok=resume)
    if not resume:
        (destination / "forests").symlink_to(Path(source_root).resolve() / "train/forests", target_is_directory=True)
    config = get_config("configs/data_gen/gen_train_data.yaml")
    config.base_dir = str(destination)
    seed_all(seed)
    provenance_path = destination / "generation_manifest.json"
    if resume:
        old = json.loads(provenance_path.read_text())
        if old["seed"] != seed or old["main_commit"] != MAIN_COMMIT or old["config"] != config:
            raise ValueError("Cannot resume generation with changed protocol/seed")
        frozen_path = destination / "crop_manifest.json"
        if frozen_path.exists() and json.loads(frozen_path.read_text()).get("status") == "verified":
            print("Frozen verified manifest found; no files rewritten. Use a new destination for regeneration.", flush=True)
            return
        # Only repair interrupted files in this experiment-owned directory.
        for folder, keys in (("forests_voxelized0.1", ("points", "labels")), ("features", ("features",)), ("occupancy", ("occupancy_grid",))):
            for path in (destination / folder).glob("*.npz"):
                try:
                    with np.load(path) as data:
                        for key in keys:
                            if not np.isfinite(data[key]).all():
                                raise ValueError("Nonfinite generated data")
                except (OSError, ValueError, EOFError, KeyError):
                    path.unlink()
    else:
        write_json(provenance_path, {"seed": seed, "config": dict(config),
                   "main_commit": MAIN_COMMIT, "directory_order": "sorted", "status": "generating"})
    # Audit each finished forest while its newly written crops are in the OS
    # page cache; avoid a second cold scan of hundreds of GB after generation.
    from tree_learn.util.data_preparation import SampleGenerator
    occupancy_seen = set()
    class AuditedGenerator(SampleGenerator):
        def get_occupancy_grid(self, occupancy_path, *pos, **kw):
            cached_first = Path(occupancy_path).exists() and self.plot_name not in occupancy_seen
            if cached_first:
                # Recreate the RNG draws skipped by main's occupancy cache path.
                n = pos[1]
                valid_count = int((self.label != kw["ignore_for_occupancy"]).sum())
                np.random.randint(0, valid_count, size=n)
            occupancy_seen.add(self.plot_name)
            return super().get_occupancy_grid(occupancy_path, *pos, **kw)

        def save(self, compressed=False):
            rows = save_and_audit_original(self, destination, seed)
            print("audited generated forest", self.plot_name, len(rows), flush=True)
    original_listdir = os.listdir
    def sorted_listdir(path):
        entries = original_listdir(path)
        return sorted(entries) if Path(path).absolute().is_relative_to(destination.absolute()) else entries
    os.listdir = sorted_listdir
    original_generator = generate_random_crops.__globals__["SampleGenerator"]
    generate_random_crops.__globals__["SampleGenerator"] = AuditedGenerator
    try:
        if workers == 1:
            generate_random_crops(config)
        else:
            generate_parallel_main(config, destination, seed, workers, AuditedGenerator)
    finally:
        os.listdir = original_listdir
        generate_random_crops.__globals__["SampleGenerator"] = original_generator
    generation = json.loads(provenance_path.read_text())
    generation["status"] = "generated"
    generation["save_workers"] = workers
    generation["crop_storage"] = "lossless NPZ; completed standard uncompressed caches preserved; new crops use deflate level 1 and write-time SHA256"
    generation["preprocessing"] = {folder: {p.name: sha256(p) for p in sorted((destination / folder).glob("*.npz"))}
                                    for folder in ("forests_voxelized0.1", "features", "occupancy")}
    write_json(provenance_path, generation)
    audit_crops(destination, seed)


def _save_planned_forest(payload):
    """A worker executes main's original save with the master's exact RNG state."""
    from tree_learn.util.data_preparation import SampleGenerator
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        obj = SampleGenerator(**payload["options"])
        for name, value in payload["attributes"].items():
            setattr(obj, name, value)
        np.random.set_state(payload["rng"])
        destination = Path(payload["destination"])
        rows = save_and_audit_original(obj, destination, payload["seed"])
    return obj.plot_name, len(rows)


def save_and_audit_original(obj, destination, seed):
    """Keep main crop extraction; audit its exact arrays before releasing them.

    Storage-only adaptation: ZIP compression is lossless and SHA is accumulated
    during writing. No cold second read of each multi-MB crop is necessary.
    """
    from tree_learn.util.data_preparation import SampleGenerator
    from .npz_io import save_npz_with_hash
    destination = Path(destination)
    original_savez, original_dump = np.savez, json.dump
    pending, rows = {}, []
    def write_arrays(path, *pos, **arrays):
        if pos:
            raise ValueError("Main crop save is expected to use named arrays")
        path = Path(path)
        digest = save_npz_with_hash(path, arrays, writer=original_savez, compression_level=1)
        points, labels = arrays["points"], arrays["instance_label"]
        if not np.isfinite(points).all():
            raise ValueError(f"Nonfinite crop {path}")
        inner = np.max(np.abs(points[:, :2]), axis=1) <= 4
        pending[path.stem] = {"path": str(path.resolve()), "sha256": digest,
            "source_forest": obj.plot_name, "generation_seed": seed, "center": arrays["center"].tolist(),
            **label_statistics(points, labels), "valid_semantic_points": int((inner & (labels != -1)).sum()),
            "valid_offset_points": int((inner & (labels > 0)).sum()), "storage": "lossless_deflate1_write_time_sha256"}
    def write_metadata(value, stream, *pos, **kw):
        original_dump(value, stream, *pos, **kw)
        stem = Path(stream.name).stem
        if stem in pending:
            stream.flush()
            row = pending.pop(stem)
            row.update(metadata=value, metadata_sha256=sha256(stream.name))
            rows.append(row)
            if len(rows) % 100 == 0:
                write_json(destination / "generation_progress" / (obj.plot_name + ".json"),
                           {"forest": obj.plot_name, "audited_crops": len(rows), "allocated": obj.n_samples_plot})
    np.savez, json.dump = write_arrays, write_metadata
    try:
        SampleGenerator.save(obj)
    finally:
        np.savez, json.dump = original_savez, original_dump
    if pending:
        raise RuntimeError("Crop metadata missing after main.save")
    rows.sort(key=lambda row: row["path"])
    folder = destination / "audited_forests"
    folder.mkdir(exist_ok=True)
    write_json(folder / (obj.plot_name + ".json"), rows)
    return rows


def generate_parallel_main(cfg, destination, seed, workers, generator_class):
    """Precompute the exact sequential main random plan, parallelize only save.

    Crop candidate generation/occupancy and np.random.choice consumption remain
    in main's sorted sequential order. Workers receive the state immediately
    before main.save, so scheduling never changes any selected crop.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    voxelized = destination / "forests_voxelized0.1"
    features = destination / "features"
    occupancy = destination / "occupancy"
    save_dir = destination / "random_crops"
    for folder in (voxelized, features, occupancy):
        if {p.stem for p in folder.glob("*.npz")} != set(FORESTS):
            raise ValueError("Parallel save requires completed main voxel/features/occupancy caches; use workers=1 first")
    def options(name):
        return {**dict(cfg.sample_generation.sample_generator), "plot_path": str(voxelized / (name + ".npz")),
                "features_path": str(features / (name + ".npz")), "save_dir": str(save_dir)}
    occupied = {}
    for name in FORESTS:
        obj = generator_class(**options(name))
        obj.get_occupancy_grid(str(occupancy / (name + ".npz")), cfg.occupancy_res,
            cfg.n_points_to_calculate_occupancy, cfg.how_far_fill, cfg.min_percent_occupied_fill, ignore_for_occupancy=-1)
        occupied[name] = np.sum(obj.occupancy_grid[:, :, 2])
    allocations = {name: int(np.round(occupied[name]/sum(occupied.values())*cfg.n_samples_total)) for name in FORESTS}
    allocations[FORESTS[-1]] += cfg.n_samples_total - sum(allocations.values())
    plan_rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = []
        for name in FORESTS:
            obj = generator_class(**options(name))
            obj.get_occupancy_grid(str(occupancy / (name + ".npz")), cfg.occupancy_res,
                cfg.n_points_to_calculate_occupancy, cfg.how_far_fill, cfg.min_percent_occupied_fill, ignore_for_occupancy=-1)
            obj.generate_candidates(cfg.n_samples_total, allocations[name], cfg.chunk_size)
            obj.check_occupancy(cfg.min_percent_occupied_choose)
            state = np.random.get_state()
            number = int(obj.filter.sum())
            selected = np.random.choice(range(number), min(allocations[name], number), replace=False)
            # These are the only RNG draws performed by main.save (filters off).
            attributes = {key: getattr(obj, key) for key in ("occupancy_res", "how_far_fill", "min_percent_occupied_fill",
                "min_percent_occupied_choose", "chunk_size", "n_samples_plot", "vertices", "rotation_angles", "centers", "filter")}
            plan_rows.append({"forest": name, "allocated": allocations[name], "valid_candidates": number,
                              "selected_valid_candidate_indices": selected.tolist()})
            cached_path = destination / "audited_forests" / (name + ".json")
            if cached_path.exists():
                cached = json.loads(cached_path.read_text())
                if len(cached) != len(selected):
                    raise ValueError("Completed forest cache count differs from the frozen selection plan")
                centers, angles = obj.centers[obj.filter][selected], obj.rotation_angles[obj.filter][selected]
                for row in cached:
                    index = int(Path(row["path"]).stem.rsplit("_", 1)[1])
                    if not np.array_equal(np.array(row["center"][:2]), centers[index]) or row["metadata"]["rotation_angle"] != angles[index]:
                        raise ValueError("Completed crop cache does not match exact main RNG replay")
                    if not Path(row["path"]).exists():
                        raise FileNotFoundError(row["path"])
                print("reused audited forest", name, len(cached), flush=True)
            else:
                futures.append(executor.submit(_save_planned_forest, {"options": options(name), "attributes": attributes,
                    "rng": state, "destination": str(destination), "seed": seed}))
            print("planned forest", name, allocations[name], flush=True)
        write_json(destination / "forest_selection_plan.json", plan_rows)
        for future in as_completed(futures):
            name, count = future.result()
            print("audited generated forest", name, count, flush=True)


def crop_record(path, destination, seed):
    path, destination = Path(path), Path(destination)
    meta_path = destination / "random_crops/json" / (path.stem + ".json")
    meta = json.loads(meta_path.read_text())
    with np.load(path) as data:
        points, labels = data["points"], data["instance_label"]
        if not np.isfinite(points).all():
            raise ValueError(f"Nonfinite points in {path}")
        row = {"path": str(path.resolve()), "sha256": sha256(path), "source_forest": meta["plot_name"],
               "generation_seed": seed, "metadata": meta, "metadata_sha256": sha256(meta_path),
               "center": data["center"].tolist(), **label_statistics(points, labels)}
        inner = np.max(np.abs(points[:, :2]), axis=1) <= 4
        # Every finite, nonempty positive instance has at least one point <=
        # min_z+0.5 under main's selected reference, so valid_offset is true.
        # This is checked against getOffset by the contract tests.
        row.update(valid_semantic_points=int((inner & (labels != -1)).sum()),
                   valid_offset_points=int((inner & (labels > 0)).sum()))
        return row


def audit_crops(destination, seed):
    destination = Path(destination)
    cached = {}
    for path in (destination / "audited_forests").glob("*.json"):
        cached.update({row["path"]: row for row in json.loads(path.read_text())})
    rows = []
    per_forest = {}
    with open(destination / "crop_records.jsonl", "w") as records:
        for path in sorted((destination / "random_crops/npz").glob("*.npz")):
            row = cached.get(str(path.resolve())) or crop_record(path, destination, seed)
            records.write(json.dumps(row, allow_nan=False) + "\n")
            # The compact manifest indexes immutable files; full statistics live in JSONL.
            rows.append({k: row[k] for k in ("path", "sha256", "source_forest", "points")})
            per_forest[row["source_forest"]] = per_forest.get(row["source_forest"], 0) + 1
            if len(rows) % 100 == 0:
                print("audited crops", len(rows), flush=True)
    manifest = {"status": "verified" if len(rows) == 25000 and set(per_forest) == set(FORESTS) else "incomplete",
                "crops": rows, "counts_by_forest": per_forest, "generation_seed": seed,
                "records_sha256": sha256(destination / "crop_records.jsonl"),
                "generation_manifest_sha256": sha256(destination / "generation_manifest.json")}
    write_json(destination / "crop_manifest.json", manifest)
    if manifest["status"] != "verified":
        raise RuntimeError("Generation did not produce the required 25,000 crops across 18 forests")
