"""`python -m backend.inference.validate <model_id>` — validate for publication.

The arch validator (`backend.arch.validate`) plus its last gate, the replay
of the analysis on the product's deployments (`analysable.replay_gate`).
`valid: true` from this entry point means the spec matches its checkpoint and
analyses on every deployment the product offers for it — or, for a model
stored without a checkpoint (`checkpoint: absent`), passes every check that
reads the spec alone and analyses on every deployment. Same arguments and
JSON as the arch CLI, which stays as the arch-only diagnostic.
"""
from __future__ import annotations

from typing import Optional

from backend.arch.validate import validate as _arch_validate
from backend.arch.validate.__main__ import main as _cli

from .analysable import replay_gate

#: The gates this entry point adds to the arch validator.
PUBLICATION_GATES = (replay_gate,)


def validate(model_id: str, spec_name: Optional[str] = None,
             overlay_dir: Optional[str] = None) -> dict:
    """The publication validate: arch gates, then the analysis replay."""
    return _arch_validate(model_id, spec_name, overlay_dir=overlay_dir,
                          downstream_gates=PUBLICATION_GATES)


def main() -> None:
    _cli(downstream_gates=PUBLICATION_GATES)


if __name__ == "__main__":
    main()
