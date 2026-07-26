# Wytham experiment configs

Ground truth:

```text
data/benchmark/wytham_vox0.1.laz
```

Pipeline input:

```text
data/pipeline/wytham/forest/wytham_vox0.1.laz
```

Create the isolated input path without copying the large file:

```bash
mkdir -p data/pipeline/wytham/forest
ln -s ../wytham_vox0.1.laz \
  data/pipeline/wytham/forest/wytham_vox0.1.laz
```

Run `pipeline_d_dual_axis_loss.yaml` first. It has `tile_generation: True`.
All other configs reuse the generated tiles and therefore use
`tile_generation: False`.

Minimal go/no-go comparison:

1. Run D and its evaluation using the already trained T2 checkpoint.
2. Run A0 with the official small-tree
   `model_weights_with_small_20241213.pth` checkpoint.
3. Continue to B/C/A ablations only if D improves the predefined primary
   metrics over A0.

Full ablation order after the go/no-go comparison:

1. B/C after T1 is available.
2. A after T0 is available.
3. F only as an optional extension experiment.

If D underperforms A0 on Wytham, run `pipeline_t2_base_only.yaml` as the only
additional diagnostic. It uses the same T2 checkpoint and base seeds as D but
sets `upper_anchor_weight: 0.0`, reducing the grouping feature from dual-anchor
4D to base-only 2D. Its evaluation config is `evaluate_t2_base_only.yaml`.

If T2 base-only remains far below A0, use the two-stage small-checkpoint
initialization:

1. `train_t2_small_warmup.yaml`: freeze the official network and train only the
   new upper-offset head for 50 epochs.
2. `train_t2_small_finetune.yaml`: unfreeze all modules and fine-tune for up to
   300 epochs with learning rate 1e-4.
3. Run `pipeline_d_small_init.yaml` using the checkpoint selected without
   looking at Wytham metrics, then evaluate with `evaluate_d_small_init.yaml`.

All benchmark runs use the same 0.1 m Wytham ground truth, grouping thresholds,
base-seed policy, evaluation thresholds, and partition definitions.

Wytham contains substantially more points than L1W, so its configs use
`max_cluster_seed_points: 1500000`. This only raises the safety guard and does
not subsample or otherwise change the seed population.
