"""Source-declared local work added by a serving execution plan.

The bound plan owns these facts; grafting and roofline readers consume them.
Integer work is deliberately not passed off as floating-point FLOPs. These
stages do not run another architecture evaluation or alter resident weights.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeStageWork:
    name: str
    read_bytes: Any
    write_bytes: Any
    source: str
    flops: Any = 0
    integer_ops: Any = 0
    compute_dtype: str | None = None
    work_basis: str = 'source_lower_bound'
    producer_uid: int | None = None
    producer_write_bytes: Any = None

    def metadata(self):
        return (
            ('runtime_stage_name', self.name),
            ('runtime_read_bytes', self.read_bytes),
            ('runtime_write_bytes', self.write_bytes),
            ('runtime_flops', self.flops),
            ('runtime_integer_ops', self.integer_ops),
            ('runtime_compute_dtype', self.compute_dtype),
            ('runtime_work_basis', self.work_basis),
            ('runtime_source', self.source),
        )
