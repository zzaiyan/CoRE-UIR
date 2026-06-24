# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
"""Shared fvcore FLOPs handlers used by benchmark and analysis scripts."""


def _shape_numel(value):
    from fvcore.nn.jit_handles import get_shape, prod

    shape = get_shape(value)
    if shape is None:
        return 0
    return int(prod(shape))


def _first_output(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def elementwise_handle(inputs, outputs):
    del inputs
    return _shape_numel(_first_output(outputs))


def zero_flop_handle(inputs, outputs):
    del inputs, outputs
    return 0


def scaled_dot_product_attention_handle(inputs, outputs):
    """Count fused SDPA as QK^T + softmax + AV."""
    from fvcore.nn.jit_handles import get_shape

    del outputs
    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])
    if q_shape is None or k_shape is None or len(q_shape) < 4:
        return 0

    batch, heads, query_len, head_dim = q_shape[:4]
    key_len = k_shape[-2]
    attention_scores = batch * heads * query_len * key_len
    qk_flops = attention_scores * head_dim
    av_flops = attention_scores * head_dim
    softmax_flops = attention_scores
    return int(qk_flops + softmax_flops + av_flops)


def native_multi_head_attention_handle(inputs, outputs):
    """Count PyTorch fused MHA as QKV projection + attention + out projection."""
    from fvcore.nn.jit_handles import get_shape

    del outputs
    query_shape = get_shape(inputs[0])
    in_proj_weight_shape = get_shape(inputs[5])
    out_proj_weight_shape = get_shape(inputs[7])
    if query_shape is None or in_proj_weight_shape is None:
        return 0

    batch, query_len, embed_dim = query_shape[:3]
    if len(query_shape) == 2:
        batch = 1
        query_len, embed_dim = query_shape
    key_len = query_len

    num_heads = int(inputs[3].toIValue())
    head_dim = embed_dim // max(num_heads, 1)

    in_proj_out_dim = in_proj_weight_shape[0]
    in_proj_flops = batch * query_len * embed_dim * in_proj_out_dim

    attention_scores = batch * num_heads * query_len * key_len
    qk_flops = attention_scores * head_dim
    av_flops = attention_scores * head_dim
    softmax_flops = attention_scores

    out_proj_flops = 0
    if out_proj_weight_shape is not None:
        out_features, in_features = out_proj_weight_shape[:2]
        out_proj_flops = batch * query_len * in_features * out_features

    return int(in_proj_flops + qk_flops + softmax_flops + av_flops + out_proj_flops)


def set_extended_flop_handles(analysis):
    """Register handlers for ops missing from fvcore's default registry."""
    return analysis.set_op_handle(
        'aten::_native_multi_head_attention',
        native_multi_head_attention_handle,
        'aten::scaled_dot_product_attention',
        scaled_dot_product_attention_handle,
        'aten::add',
        elementwise_handle,
        'aten::mul',
        elementwise_handle,
        'aten::div',
        elementwise_handle,
        'aten::sigmoid',
        elementwise_handle,
        'aten::softmax',
        elementwise_handle,
        'aten::gelu',
        elementwise_handle,
        'aten::embedding',
        zero_flop_handle,
        'aten::unflatten',
        zero_flop_handle,
    )


__all__ = [
    'elementwise_handle',
    'native_multi_head_attention_handle',
    'scaled_dot_product_attention_handle',
    'set_extended_flop_handles',
    'zero_flop_handle',
]
