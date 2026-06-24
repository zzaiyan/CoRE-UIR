# CoRE-UIR

Code for **CoRE-UIR: Prior-Guided Common and Residual Experts for Efficient
All-in-One Remote Sensing Image Restoration**.

CoRE-UIR uses a frozen degradation prior embedding (DPE) to guide an image
restoration backbone for single and compound remote-sensing degradations.

## Installation

```bash
conda activate torch25
pip install -r requirements.txt
```

The DPE module uses `openai/clip-vit-base-patch32` from HuggingFace
Transformers. It will be downloaded automatically if it is not already cached.

## Data

Edit `configs/core_uir_mdvd.yml` and
`configs/dpe_clip_b32_multilabel_mdvd.yml` so `/path/to/MDVD-108K` points to
your dataset root.

Expected layout:

```text
MDVD-108K/
  train/
    fog/{input,gt}
    dust/{input,gt}
    rain/{input,gt}
    lowlight/{input,gt}
    motion/{input,gt}
    defocus/{input,gt}
    fog-lowlight/{input,gt}
    fog-motion/{input,gt}
    dust-lowlight/{input,gt}
    dust-motion/{input,gt}
    rain-lowlight/{input,gt}
    rain-motion/{input,gt}
  test/
    ...
```

## Checkpoints

Large weights are not tracked by git. Place them under `pretrained/`:

```text
pretrained/dpe_clip_b32_multilabel_mdvd.ckpt
pretrained/core_uir_mdvd.pth
```

If needed, convert an older restoration checkpoint:

```bash
python tools/convert_core_uir_checkpoint.py \
  path/to/old_net_g.pth pretrained/core_uir_mdvd.pth
```

## Train DPE

```bash
python train_dpe.py -opt configs/dpe_clip_b32_multilabel_mdvd.yml --gpu 0
```

After training, copy or symlink the DPE checkpoint:

```bash
mkdir -p pretrained
ln -s ../prior/checkpoints/router_mdvd_clip-b-32_multilabel/last.ckpt \
  pretrained/dpe_clip_b32_multilabel_mdvd.ckpt
```

## Train CoRE-UIR

```bash
python train.py -opt configs/core_uir_mdvd.yml --gpu 0
```

## Test

```bash
python test.py -opt configs/core_uir_mdvd.yml \
  --weights pretrained/core_uir_mdvd.pth --gpu 0
```

Results are written to `results/CoRE-UIR_MDVD/`.

## Benchmark

```bash
python benchmark.py -opt configs/core_uir_mdvd.yml \
  --weights pretrained/core_uir_mdvd.pth --gpu 0
```

Benchmark reports restoration, DPE, and end-to-end parameters, FLOPs, speed,
and memory.
