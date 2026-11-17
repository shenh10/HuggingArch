"""Absorbed MLA multiplies the serving strided activation, not a copy."""
import pytest
torch=pytest.importorskip('torch')
from kernel_bench.providers.sglang_mla import _absorbed_bmm_build

@pytest.mark.parametrize('batch,query,heads', [(1,1,96),(2,3,12)])
@pytest.mark.parametrize('width,output,padded', [(128,512,192),(512,128,512)])
def test_serving_bmm_strides_and_values(batch,query,heads,width,output,padded):
    gen=torch.Generator().manual_seed(67)
    # Small integer products/sums make exact equality a layout check across
    # CPU BLAS implementations, independent of BF16 rounding order.
    inputs={'x':torch.randint(-1,2,(batch,heads,query,width),generator=gen).to(torch.bfloat16),
            'w':torch.randint(-1,2,(heads,output,width),generator=gen).to(torch.bfloat16)}
    case=_absorbed_bmm_build(padded)(None,None,inputs)
    a,w=case.run.args
    assert a.shape==(heads,batch*query,width)
    assert a.stride()==(padded,heads*padded,1)
    assert w.shape==(heads,width,output) and w.is_contiguous()
    actual=case.to_canonical(case.run())
    expected=torch.einsum('bhsk,hnk->bhsn',inputs['x'],inputs['w'])
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
