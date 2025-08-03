import torch
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTConfig
from torch_int.models.opt import Int8OPTDecoderLayer
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from typing import Tuple
from icecream import ic
from functools import partial


def store_act(module, x, y, act_dict, name):
    # x and y from hooks are tuples/lists of inputs/outputs
    x0 = x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x
    y0 = y[0] if isinstance(y, (list, tuple)) and len(y) > 0 else y
    # Only store valid tensor pairs
    if isinstance(x0, torch.Tensor) and isinstance(y0, torch.Tensor) and name is not None:
        act_dict[name] = (x0, y0)


@torch.no_grad()
def test_opt_decoder_layer():
    config = OPTConfig.from_pretrained('facebook/opt-125m')
    # Ensure HF attention implementation is set; otherwise HF raises KeyError(None)
    # "eager" is universally available. Set both public and private to handle HF versions.
    setattr(config, "attn_implementation", getattr(config, "attn_implementation", None) or "eager")
    setattr(config, "_attn_implementation", getattr(config, "_attn_implementation", None) or "eager")
    B, L, D, H = 1, 256, config.hidden_size, config.num_attention_heads

    x = torch.randn(B, L, D)
    layer = OPTDecoderLayer(config)
    layer.eval()
    act_dict = {}
    # Only hook the exact linear modules we care about
    target_names = {
        'self_attn.q_proj',
        'self_attn.k_proj',
        'self_attn.v_proj',
        'self_attn.out_proj',
        'fc1',
        'fc2',
    }
    for name, module in layer.named_modules():
        if name in target_names and isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name)
            )
    y = layer(x)[0]

    attn_input_scale = act_dict['self_attn.q_proj'][0].abs().max() / 127
    q_output_scale = act_dict['self_attn.q_proj'][1].abs().max() / 127
    k_output_scale = act_dict['self_attn.k_proj'][1].abs().max() / 127
    v_output_scale = act_dict['self_attn.v_proj'][1].abs().max() / 127
    out_input_scale = act_dict['self_attn.out_proj'][0].abs().max() / 127
    fc1_input_scale = act_dict['fc1'][0].abs().max() / 127
    fc2_input_scale = act_dict['fc2'][0].abs().max() / 127
    int8_layer = Int8OPTDecoderLayer.from_float(
        layer, attn_input_scale, q_output_scale, k_output_scale, v_output_scale, out_input_scale, fc1_input_scale, fc2_input_scale).cuda()
    int8_layer.eval()

    y_hat = int8_layer(x.cuda())[0].cpu()

    # # ic(y_hat)
    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    test_opt_decoder_layer()
