"""Train or evaluate the Phase-I Degradation Prior Embedding."""

import sys

from prior.train_router_common import parse_args_with_config, run_training


if __name__ == '__main__':
    if '-opt' in sys.argv:
        sys.argv[sys.argv.index('-opt')] = '--config'
    args = parse_args_with_config(
        default_config='configs/dpe_clip_b32_multilabel_mdvd.yml')
    run_training(args)
