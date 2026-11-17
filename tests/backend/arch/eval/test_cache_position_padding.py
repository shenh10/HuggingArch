"""Position ownership padding has one storage law across capacity readers."""
from dataclasses import replace

import sympy as sp

from backend.arch.cache_formats import CacheFormat
from backend.arch.eval.cache import Cache, CachedTensor
from backend.arch.eval.cache_view import CacheClass, CacheView
from backend.arch.eval.values import CACHE_LEN


def tensor(**kwargs):
    return CachedTensor(node=1, name='key', elements=128, inner=128, rows=1,
                        format=CacheFormat.parse({'dtype': 'BF16'}, 'test'),
                        positions_over=8, **kwargs)


def test_ceil_position_ownership_agrees_with_breakdown_and_formula():
    t = tensor(position_padding=True)
    view = CacheView(entries=(CacheClass(Cache(component=1, tensors=(t,)), 3),))
    assert t.stored == CACHE_LEN  # graph identity remains the declared law
    assert float(t.bytes_at(17)) == 3 * 256
    assert float(t.bytes_at(0)) == 0
    assert view.storage_bytes_at(17) == 3 * 3 * 256
    assert view.storage_rows(17)[0]['request_bytes'] == view.storage_bytes_at(17)
    assert view.request_formula(17)['result_bytes'] == view.storage_bytes_at(17)
    assert view.growth_and_floor() == (3 * 256 / 8, 0)
    # Read selection is independent of where the persistent final slot lands.
    assert float(replace(t, read=sp.Min(CACHE_LEN, 8)).bytes_at(17, read=True)) == 256


def test_default_partition_and_recurrent_state_keep_existing_storage():
    assert tensor().bytes_at(17) == 17 * 256 / 8
    recurrent = replace(tensor(position_padding=True), recurrent=True,
                        stored=1, read=1, positions_over=1)
    assert recurrent.bytes_at(17) == recurrent.bytes_at(1024) == 256


def test_bounded_cache_pads_its_window_not_the_whole_context():
    bounded = tensor(position_padding=True, stored=sp.Min(CACHE_LEN, 10))
    view = CacheView(entries=(CacheClass(Cache(component=1, tensors=(bounded,)), 1),))
    assert float(bounded.bytes_at(100)) == 2 * 256
    assert view.growth_and_floor() == (0, 2 * 256)
