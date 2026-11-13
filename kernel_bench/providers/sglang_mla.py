"""SGLang MLA absorbed BF16 BMMs with serving's physical operand strides.

Audited at SGLang 1b2870d, deepseek_common/attention_forward_methods/
forward_mla.py:611,970. The Q input is the NoPE slice of a full 192-wide
head; V starts as contiguous latent heads. Weight transposition/repacking
is load-time setup. DCP8 uses 96 heads, ordinary TP8 uses 12.
"""
from . import BenchCase, KernelSignature, Launch, Provider, register


def _absorbed_bmm_build(padded_width):
    def build(inv, env, inputs):
        import torch
        x, w = inputs['x'], inputs['w']
        batch, heads, query, width = x.shape
        storage = torch.empty(batch * query, heads, padded_width,
                              dtype=torch.bfloat16, device=x.device)
        storage[..., :width] = x.permute(0,2,1,3).reshape(batch*query,heads,width)
        activation = storage[..., :width].transpose(0,1)
        # SGLang prepares w_kc / w_vc as [heads,reduce,out] contiguous.
        weight = w.transpose(1,2).to(torch.bfloat16).contiguous()
        return BenchCase(run=Launch(torch.bmm,activation,weight),
            to_canonical=lambda out:out.reshape(heads,batch,query,-1).permute(1,0,2,3))
    return build


for name, reduce, output, padded in [('q',128,512,192),('v',512,128,512)]:
    register(Provider(library='torch',signature='per_head_gemm',
        kernel_impl=f'torch_sglang_mla_absorb_{name}_bmm',import_name='torch',
        serves=KernelSignature(signature='per_head_gemm',weight=('bf16',),
            activation=('bf16',),output=('bf16',),
            enum=(('d_reduce',(reduce,)),('d_out',(output,)),('n_heads',(12,96)))),
        build=_absorbed_bmm_build(padded)))
