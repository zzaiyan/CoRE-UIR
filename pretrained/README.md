# Pretrained Weights

Large checkpoints are intentionally not tracked by git. Place released weights
here before testing or benchmarking.

Local files remain supported, while public configurations can resolve the same
weights directly from `hf://zzaiyan/CoRE-UIR/<filename>`.

Expected files:

- `pretrained/dpe_clip_b32_mdvd.ckpt`
  - Phase-I multi-label DPE adapter for MDVD-108K.
  - Contains only the trained adapter and classification head. The frozen
    `openai/clip-vit-base-patch32` backbone is loaded from Hugging Face.
- `pretrained/core_uir_mdvd.pth`
  - Phase-II CoRE-UIR restoration checkpoint for MDVD-108K.
- `pretrained/dpe_clip_b32_mdrs.ckpt`
  - Phase-I single-label DPE adapter for MDRS-Landsat.
  - Contains only the trained adapter and classification head. The frozen
    `openai/clip-vit-base-patch32` backbone is loaded from Hugging Face.
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
