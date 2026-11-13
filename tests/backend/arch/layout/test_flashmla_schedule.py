from backend.arch.layout.flashmla_schedule import native_head_partitions, windowed_schedule


def test_native_head_partitions_match_installed_wheel_probe():
    assert native_head_partitions(148, 1, 96) == 74
    assert native_head_partitions(148, 73, 96) == 74
    assert native_head_partitions(148, 74, 96) == 148
    assert native_head_partitions(148, 128, 96) == 148


def test_windowed_schedule_has_split_and_no_work_combine_cases():
    assert windowed_schedule((256,), 74).num_splits == (4,)
    assert windowed_schedule((2048,), 74).num_splits == (32,)
    assert windowed_schedule((0,), 74).num_splits == (1,)
    assert windowed_schedule((64,), 74).num_splits == (1,)
    schedule = windowed_schedule((256,) * 128, 148)
    assert all(1 <= count <= 4 for count in schedule.num_splits)


def test_uniform_work_matches_independent_source_schedule():
    from backend.arch.layout.flashmla_schedule import windowed_work
    work = windowed_work(1, (256,), 74)
    assert work.total_splits == 4
    assert work.combining_rows == 1
    assert work.combining_splits == 4
    skipped = windowed_work(1, (64,), 74)
    assert skipped.total_splits == 1
    assert skipped.combining_rows == skipped.combining_splits == 0
    multi = windowed_work(16, (256,), 74)
    actual = windowed_schedule((256,) * 16, 74).num_splits
    assert multi.total_splits == sum(actual)
    assert multi.combining_rows == sum(n > 1 for n in actual)
    assert multi.combining_splits == sum(n for n in actual if n > 1)


def test_symbolic_periodic_work_reads_same_numeric_scheduler():
    import sympy as sp
    from backend.arch.layout.flashmla_schedule import windowed_work
    batch = sp.Symbol('batch', integer=True, positive=True)
    length = sp.Symbol('length', integer=True, positive=True)
    formula = windowed_work(batch, (length,), 7)
    for b, n in ((1, 64), (1, 256), (3, 256), (8, 2048)):
        actual = windowed_schedule((n,) * b, 7).num_splits
        bindings = {batch: sp.Integer(b), length: sp.Integer(n)}
        assert formula.total_splits.xreplace(bindings) == sum(actual)
        assert formula.combining_rows.xreplace(bindings) == sum(k > 1 for k in actual)
        assert formula.combining_splits.xreplace(bindings) == sum(k for k in actual if k > 1)


def test_runtime_law_serialization_and_compiled_readers():
    import numpy as np
    import sympy as sp
    from backend.arch.law_text import read_walk_law
    from backend.arch.layout.flashmla_schedule import windowed_work
    from backend.measure.auto_sweep import compile_law, compile_law_float, compile_law_vector
    batch, length = sp.symbols('batch length', integer=True, positive=True)
    work = windowed_work(batch, (length,), 74)
    laws = (work.total_splits, work.combining_rows, work.combining_splits)
    assert max(len(str(law)) for law in laws) < 100
    restored = tuple(read_walk_law(str(law)) for law in laws)
    for compiler in (compile_law, compile_law_float):
        assert list(compiler(restored, ('batch', 'length'))(1, 256)) == [4, 1, 4]
        assert list(compiler(restored, ('batch', 'length'))(1, 64)) == [1, 0, 0]
    actual = compile_law_vector(restored, ('batch', 'length'))(
        np.array([1, 1]), np.array([256, 64]))
    assert [item.tolist() for item in actual] == [[4, 1], [1, 0], [4, 0]]


def test_periodic_runtime_counts_match_independent_oracle():
    from backend.arch.runtime_laws import windowed_counts
    for batch in (1, 2, 16, 128):
        for pattern in ((0,), (64,), (256,), (1, 1, 1, 2, 2), (64, 65, 128, 129, 2048)):
            parts = native_head_partitions(148, batch * len(pattern), 96)
            actual = windowed_schedule(pattern * batch, parts).num_splits
            assert windowed_counts(batch, parts, pattern) == (
                sum(actual), sum(n > 1 for n in actual), sum(n for n in actual if n > 1))


def test_windowed_precompute_admission_rejects_native_fallback():
    import pytest
    import sympy as sp
    from backend.arch.layout.flashmla_schedule import windowed_work
    from backend.arch.runtime_laws import windowed_counts
    assert windowed_counts(2220, 148, (256,))[0] >= 2220
    with pytest.raises(ValueError, match='48KiB admission'):
        windowed_counts(2221, 148, (256,))
    b = sp.Symbol('batch', integer=True, positive=True)
    law = windowed_work(b, (256,), 148).total_splits
    with pytest.raises(ValueError, match='48KiB admission'):
        law.xreplace({b: sp.Integer(2221)})
