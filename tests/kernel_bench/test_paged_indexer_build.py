"""Exact paged FP8 indexer contract: separate pages/requests and scale tails."""
import sys
import types
import pytest

torch = pytest.importorskip('torch')
from kernel_bench.invocation import Invocation
from kernel_bench.providers import indexer

@pytest.mark.parametrize('length', [65, 257])
@pytest.mark.parametrize('format', ['fp8_group128', 'fp8_pow2_group128'])
def test_paged_indexer_layout(monkeypatch, length, format):
    calls = []
    def metadata(lens, page, sms):
        assert lens.shape == (2, 1) and lens.dtype == torch.int32
        assert page == 64 and sms == 148
        calls.append('schedule')
        return lens.clone()
    def logits(q, cache, w, lens, table, schedule, maximum, clean_logits):
        assert clean_logits is False
        assert q.shape == (2, 1, 32, 128) and q.dtype == torch.float8_e4m3fn
        assert cache.dtype == torch.uint8 and cache.shape[1:] == (64, 1, 132)
        assert table.dtype == torch.int32 and w.dtype == torch.float32
        flat = cache.flatten(1)
        keys = flat[:, :8192].contiguous().view(torch.float8_e4m3fn).float().reshape(-1,64,128)
        scales = flat[:, 8192:].contiguous().view(torch.float32)
        if format == 'fp8_pow2_group128':
            assert torch.allclose(scales.log2(), scales.log2().round())
        keys = (keys * scales[...,None])[table.long()].reshape(2,maximum,128)
        return torch.einsum('bhd,bjd->bhj',q[:,0].float(),keys).relu().mul(w[:,:,None]).sum(1)
    monkeypatch.setattr(indexer, '_require_cuda', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: types.SimpleNamespace(multi_processor_count=148))
    monkeypatch.setitem(sys.modules, 'deep_gemm',types.SimpleNamespace(get_paged_mqa_logits_metadata=metadata,fp8_paged_mqa_logits=logits))
    inv=Invocation(signature='indexer_logits',op='indexer_logits',phase='decode',tp=1,
        dims={'B':2,'s_query':1,'s_kv':length,'n_indexer_heads':32,'d_indexer_head':128,'kv_block_size':64},
        dtype_env={'activation':'fp8_e4m3','output':'fp32','kv_format':format})
    gen=torch.Generator().manual_seed(23)
    inputs={'q':torch.randn(2,32,128,generator=gen,dtype=torch.bfloat16),
            'k':torch.randn(2,length,128,generator=gen,dtype=torch.bfloat16),
            'weights':torch.rand(2,32,generator=gen)}
    case=indexer._deepgemm_indexer_build(inv,None,inputs)
    out=case.to_canonical(case.run())
    ref=torch.einsum('bhd,bjd->bhj',inputs['q'].float(),inputs['k'].float()).relu().mul(inputs['weights'][:,:,None]).sum(1)
    assert out.shape==(2,length)
    assert ((out-ref).abs().mean()/ref.abs().mean())<0.02
    assert calls==['schedule']

@pytest.mark.parametrize('change', [
    {'s_query':2}, {'n_indexer_heads':64}, {'d_indexer_head':64},
    {'kv_block_size':128}, {'B':129}, {'phase':'prefill'},
    {'activation':'bf16'}, {'output':'bf16'}, {'kv_format':'plain'},
])
def test_paged_indexer_closed_domain(change):
    from dataclasses import replace
    from kernel_bench.providers import all_providers
    provider=next(p for p in all_providers() if p.kernel_impl=='deepgemm_fp8_paged_mqa_logits')
    dims={'B':1,'s_query':1,'s_kv':257,'n_indexer_heads':32,'d_indexer_head':128,'kv_block_size':64}
    dtypes={'activation':'fp8_e4m3','kv':'fp8_e4m3','output':'fp32','kv_format':'fp8_group128'}
    inv=Invocation(signature='indexer_logits',op='indexer_logits',phase='decode',tp=8,dims=dims,dtype_env=dtypes)
    assert provider.serves.accepts(inv)
    for key,value in change.items():
        if key in dims: inv=replace(inv,dims={**dims,key:value})
        elif key=='phase': inv=replace(inv,phase=value)
        else: inv=replace(inv,dtype_env={**dtypes,key:value})
    assert not provider.serves.accepts(inv)

@pytest.mark.parametrize('format', ['fp8_group128', 'fp8_pow2_group128'])
def test_byte_storage_gate_requires_exact_declared_record(format):
    from kernel_bench.launch_trace import Operand,Call,Trace,disagreement,paged_byte_cache_shape
    shape=paged_byte_cache_shape((2,257,128),64,format)
    assert shape==(10,64,1,132)
    def gate(actual,declared=()):
        trace=Trace(calls=[Call('paged', (Operand('uint8',actual),))],saw_result=True)
        return disagreement(trace,declared_dtypes=['fp8_e4m3','fp32'],
            declared_elements=[2*257*128],packed_byte_shapes=declared)
    assert gate(shape) is not None
    assert gate(shape,(shape,)) is None
    assert gate((10,64,1,128),(shape,)) is not None
    assert gate((20,64,1,132),(shape,)) is not None
