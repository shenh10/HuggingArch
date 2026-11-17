"""Runtime schedulers cannot reuse a walk across different SM counts."""
from dataclasses import replace
from unittest.mock import patch

from backend.inference.framework_bindings import (
    ServingDeployment, card_class,
)
from backend.inference.gpu_specs import get_gpu


def test_b300_sm_count_comes_from_the_audited_device_not_peak_flops():
    # torch.cuda.get_device_properties on the B300 used for the source probe.
    deployment = ServingDeployment.at(gpu_sku='B300', activation_dtype='BF16')
    assert deployment.sm_count == 148
    assert ServingDeployment.at(gpu_sku=None, activation_dtype='BF16').sm_count is None


def test_sm_count_changes_card_class_even_at_the_same_compute_capability():
    original = get_gpu('B300')
    cards = {name: replace(original, sku=name, sm_count=count)
             for name, count in (('fixture-full', 148), ('fixture-partial', 132))}
    with patch('backend.inference.framework_bindings.get_gpu', side_effect=cards.__getitem__):
        assert card_class(cards['fixture-full']) != card_class(cards['fixture-partial'])


def test_unknown_sm_count_stays_distinct_from_the_observed_card():
    original = get_gpu('B300')
    unknown = replace(original, sm_count=None)
    assert card_class(original) != card_class(unknown)
    assert ('sm_count', None) in card_class(unknown).runtime
