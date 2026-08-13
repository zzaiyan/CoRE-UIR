# Pretrained Weights

Large checkpoints are intentionally not tracked by git. Place released weights
here before testing or benchmarking.

Expected files:

- `pretrained/dpe_clip_b32_multilabel_mdvd.ckpt`
  - Phase-I DPE checkpoint.
  - In the development tree this corresponds to
    `prior/checkpoints/router_mdvd_clip-b-32_multilabel/last.ckpt`.
- `pretrained/core_uir_mdvd.pth`
  - Phase-II CoRE-UIR restoration checkpoint for MDVD-108K.
- `pretrained/dpe_clip_b32_mdrs.ckpt`
  - Phase-I single-label DPE checkpoint for MDRS-Landsat.
  - In the development tree this corresponds to
    `prior/checkpoints/router_mdrs_clip-b-32/last.ckpt`.
- `pretrained/core_uir_mdrs.pth`
  - Phase-II CoRE-UIR restoration checkpoint for MDRS-Landsat.

Example:

```bash
python test.py -opt configs/core_uir_mdvd.yml --weights pretrained/core_uir_mdvd.pth
```

For MDRS-Landsat:

```bash
python test.py -opt configs/core_uir_mdrs.yml --weights pretrained/core_uir_mdrs.pth
```
