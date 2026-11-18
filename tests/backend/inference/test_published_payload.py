"""The published-column declaration replaces each consumer's name list.

The expected literals below are the lists as they stood before the declaration.
They are copied once on purpose: asking the declaration for both sides would
agree with itself after any edit.
"""
from __future__ import annotations

import pytest

from backend.inference.capacity import _PHASE_LAUNCH_TOTALS
from backend.inference.module_rows import _TIMING_DROP
from backend.inference.row_roofline import (ROW, Share, Column,
                                            POINT_ADDITIVE_COLUMNS,
                                            column_names)


ROW_ADDS = (
    "flops_per_rank", "access_bytes_per_rank", "act_in_bytes_per_rank",
    "kv_read_bytes_per_rank", "act_out_bytes_per_rank",
    "convert_bytes_per_rank", "weight_read_bytes_per_rank",
    "metadata_bytes_per_rank", "comm_bytes_per_rank", "comm_tensor_bytes",
    "t_compute_s", "t_memory_s", "t_comm_s", "t_launch_s", "t_sol_s",
    "t_op_s",
)

PHASE_ADDS = (
    "flops_per_step", "mem_read_bytes", "mem_write_bytes", "total_bytes",
    "t_compute_s", "t_memory_s", "t_comm_s", "t_sol_s",
)

MODULE_DROPS = frozenset({
    "node_dims", "shape_in_exact", "shape_in_dims", "shape_out_dims",
    "weight_shards", "shape_json", "dtype_env_json", "shape_weight",
    "shape_weight_dims", "weight_bytes_per_rank",
    "weight_bytes_total_per_rank",
    "weight_bytes_per_token_active_per_rank", "weight_read_bytes_per_rank",
    "metadata_bytes_per_rank", "metadata_bytes_total_per_rank",
    "act_in_bytes_per_rank", "kv_read_bytes_per_rank",
    "act_out_bytes_per_rank", "peak_flops_per_sec_for_op", "act_dtype",
    "moe_block_member", "active_ratio_per_token", "derived_group",
    "count_per_step", "parent_aggregation_id", "comm_op",
    "comm_tensor_bytes", "comm_base_stride", "comm_route_fanout",
    "comm_collective_in_shape",
    "comm_collective_out_shape", "comm_payload_shape", "comm_payload_role",
    "comm_payload_dtype_bytes", "comm_payload_scale", "point_law",
    "zero_cost",
})


def test_the_row_columns_that_add_are_the_ones_that_added():
    assert POINT_ADDITIVE_COLUMNS == ROW_ADDS
    assert column_names(published_by=ROW, additive=True) == ROW_ADDS


def test_the_phase_launch_totals_are_derived_from_the_declaration():
    assert _PHASE_LAUNCH_TOTALS == PHASE_ADDS


def test_the_module_table_drops_the_fields_it_dropped():
    assert frozenset(_TIMING_DROP) == MODULE_DROPS


def test_a_column_is_held_to_the_share_its_name_states():
    with pytest.raises(ValueError, match="its name states 'rank'"):
        Column("widget_bytes_per_rank", share=Share.DEPLOYMENT)
    with pytest.raises(ValueError, match="its name states 'group'"):
        Column("throughput.per_group", share=Share.DEPLOYMENT)
    # A hand-read producer name with no suffix is the reason `share` exists.
    Column("flops_per_step", share=Share.RANK)


def test_a_row_column_reads_a_real_point_field():
    with pytest.raises(ValueError, match="no field of the row at the point"):
        Column("widget_per_rank", source="widget", share=Share.RANK)
    with pytest.raises(ValueError, match="no field of the row's times"):
        Column("widget_t_s", source="times.widget")


def test_there_is_no_work_field():
    """`share` and `additive` answer the two aggregation questions.

    A flattened `work` enum would mix schedule granularity, model dimensions
    and units; those belong to the laws and the dimension registry, not to a
    published column.
    """
    import backend.inference.row_roofline as row_roofline

    assert not hasattr(row_roofline, "Work")
    assert tuple(Column.__dataclass_fields__) == (
        "name", "published_by", "source", "share", "additive",
        "on_module_table",
    )
