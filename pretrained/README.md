# Pretrained Weights

Large checkpoints are intentionally not tracked by git. Place released weights
here before testing or benchmarking.

Expected files:

- `pretrained/dpe_clip_b32_multilabel_mdvd.ckpt`
  - Phase-I DPE checkpoint.
  - In the development tree this corresponds to
    `prior/checkpoints/router_mdvd_clip-b-32_multilabel/last.ckpt`.
- `pretrained/core_uir_mdvd.pth`
  - Optional Phase-II CoRE-UIR restoration checkpoint.

Example:

```bash
python test.py -opt configs/core_uir_mdvd.yml --weights pretrained/core_uir_mdvd.pth
```
