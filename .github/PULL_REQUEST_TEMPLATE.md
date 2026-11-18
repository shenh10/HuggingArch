<!-- Thanks for contributing! Keep PRs small and focused. -->

## What & why

<!-- What does this change, and why? Link any related issue (e.g. Closes #123). -->

## Type

- [ ] New model spec (`backend/arch/models/`)
- [ ] New component (`backend/arch/components/library/`)
- [ ] Bug fix
- [ ] Feature
- [ ] Docs / chore

## Checklist

- [ ] New model spec: validator output below (`valid: true`); CI runs the spec gate
- [ ] Code change: `uv run pytest tests/` passes
- [ ] `uv run ruff check .` is clean (code changes)
- [ ] No new import of `backend/service/` from core (boundary test still passes)
- [ ] Auth/admin changes add a **new** numbered migration (no edits to existing ones)
- [ ] Schema / retired-field change (`backend/arch/schema/**`, `backend/arch/spec_parsing.py`): ran `PYTHONPATH=. ./.venv/bin/python tools/check_stored_specs.py` (use `--local-only` or `--db-json` when the live DB is unreachable; default is fail-closed)

## For a new model

<!-- Paste the validator output; it should say valid: true, diff: 0
     (or explain a documented residual diff). -->

```
$ python -m backend.inference.validate <model-id>
valid: true, diff: 0
```
