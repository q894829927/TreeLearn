# Upstream source provenance

* ForestFormer3D: https://github.com/SmartForest-no/ForestFormer3D
  revision `6a75c3735e4a4108d02ee944a8b93177f2360a4f`.
  `spconv_unet.py` is the full five-level configurable sparse U-Net source.
  `decoder.py` contains CrossAttentionLayer, SelfAttentionLayer, FFN and
  ForAINetv2QueryDecoder_XAwarequery from `oneformer3d/query_decoder.py`.
  Removed OpenMMLab registration; BaseModule becomes nn.Module; unused attention
  weights are disabled. Architecture and parameter names are retained.
  License: **CC BY-NC 4.0**, reproduced in `FORESTFORMER_LICENSE`.
  Original U-Net attribution to sunjiahao1999/SPFormer is retained in its source.
* TripleMixer: https://github.com/Grandzxw/TripleMixer
  revision `fcafe56f4f224a24ec566539b6731d9d2c700b1f`.
  `geometry.py` retains the original GeoMixer implementation and copyright.
  License: Apache 2.0, see `TRIPLEMIXER_LICENSE` and source header.
* P3Former: https://github.com/InternRobotics/P3Former
  inspected revision `2b8faa2b0b168d78713092b223a4a6e8ba63e44e`,
  `p3former/decode_heads/p3former_head.py`, `_Masked_Focal_Attention`.
  No upstream file is vendored. `../modules.py` independently implements the
  mask-region sum and dynamic input/update gating, with empty-region fallback,
  residual blending, and native torch LayerNorm/GELU. It is an adaptation of the
  update mechanism, not the full P3Former method or a verbatim upstream module.

The surrounding raw-coordinate data pipeline, binary semantic adaptation,
native-torch ISA sampling, losses, runner, and disk-backed inference integration
are project adaptations. This is a trainable combined architecture, **not** an
unchanged official ForestFormer3D reproduction. Full source and experiment
configuration hashes are stored in checkpoints and evaluation reports.
