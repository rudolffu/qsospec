"""Contracts for exact evaluator reuse and workflow-local conservative seeds."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from qsospec import FitPerformanceConfig, GlobalContinuumConfig, Spectrum
from qsospec.complex_recipes import ComponentRecipe, ComplexRecipe
from qsospec.fitting.complexes import GenericComplexContext, fit_generic_complex
from qsospec.fitting.multistart import solve_multistart
from qsospec.fitting.performance import (
    ExactEvaluationCache,
    current_session,
    line_seed_key,
    performance_scope,
)


def example(profile="gaussian"):
    recipe = ComplexRecipe(
        "cache_test",
        (),
        "cache",
        (4900.0, 5100.0),
        ((4900.0, 5100.0),),
        (),
        (
            ComponentRecipe(
                "line",
                ("oiii_5008",),
                "narrow",
                profile=profile,
                velocity_bounds_kms=(-1000.0, 1000.0),
                fwhm_bands_kms=((70.0, 1200.0),),
            ),
        ),
        (),
        continuum_mode="residual_linear",
    )
    context = GenericComplexContext(recipe, ["line"], 100.0)
    wave = np.linspace(4900.0, 5100.0, 220)
    theta = context.initial.copy()
    theta[context.index["line.flux"]] = 40.0
    spectrum = Spectrum.from_arrays(
        wave, context.model(theta, wave), err=np.full_like(wave, 0.1), wave_frame="rest", flux_unit="relative"
    )
    continuum = SimpleNamespace(model=np.zeros_like(wave), param_values={}, wave_rest=wave)
    return recipe, context, spectrum, continuum


def test_config_validation():
    for kwargs in (
        {"warm_starts": 1},
        {"cache_evaluations": "yes"},
        {"evaluation_cache_max_bytes": -1},
        {"evaluation_cache_max_bytes": True},
    ):
        with pytest.raises(ValueError):
            FitPerformanceConfig(**kwargs)
    assert GlobalContinuumConfig().performance.evaluation_cache_max_bytes == 16 * 1024 * 1024


def test_exact_cache_immutable_identity_eviction_and_cap():
    calls = []
    wave = np.arange(5.0, dtype=float)
    definition = [0]

    def evaluate(theta, derivatives):
        calls.append((theta.copy(), derivatives))
        return np.ones((5, 2)) * theta[0], (np.ones((5, 2)),) if derivatives else None

    cache = ExactEvaluationCache(evaluate, wave, state_token=lambda: definition[0])
    first = cache(np.array([1.0]), False)
    assert first[1] is None
    full = cache(np.array([1.0]), True)
    assert len(calls) == 1 and cache.statistics["hits"] == 1
    with pytest.raises(ValueError):
        full[0][0, 0] = 3.0
    with pytest.raises(ValueError):
        full[1][0][0, 0] = 3.0
    cache(np.array([np.nextafter(1.0, 2.0)]), False)
    cache(np.array([1.0]), False)
    assert len(calls) == 3
    wave[0] = -0.5
    cache(np.array([1.0]), False)
    definition[0] = 1
    cache(np.array([1.0]), False)
    assert len(calls) == 5
    cache(np.array([-0.0]), False)
    cache(np.array([0.0]), False)
    assert len(calls) == 7
    cache.config = FitPerformanceConfig(evaluation_cache_max_bytes=1)
    cache(np.array([2.0]), False)
    assert cache.entry is None and cache.statistics["bypasses"] == 1


def test_cache_disabled_and_two_point_upgrade():
    evaluate = lambda theta, need: (np.ones((4, 2)), (np.ones((4, 2)),) if need else None)
    cache = ExactEvaluationCache(evaluate, np.arange(4.0), derivatives=False)
    assert cache(np.array([1.0]), False)[1] is None
    assert cache(np.array([1.0]), True)[1] is not None
    assert cache.statistics["misses"] == 2
    cache.config = FitPerformanceConfig(cache_evaluations=False)
    cache.entry = None
    cache(np.array([1.0]), False)
    cache(np.array([1.0]), True)
    assert cache.entry is None and cache.statistics["misses"] == 4


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian"])
def test_generic_cache_exact_full_result(profile):
    recipe, _, spectrum, continuum = example(profile)
    cold = fit_generic_complex(spectrum, continuum, recipe, performance=FitPerformanceConfig(cache_evaluations=False))
    cached = fit_generic_complex(spectrum, continuum, recipe, performance=FitPerformanceConfig())
    for key in ("param_values", "param_errors", "metrics", "metric_errors"):
        a, b = getattr(cold, key), getattr(cached, key)
        assert a.keys() == b.keys()
        for name in a:
            assert np.array_equal(a[name], b[name], equal_nan=True)
    for key in ("model", "covariance", "fit_mask"):
        assert np.array_equal(getattr(cold, key), getattr(cached, key), equal_nan=True)
    assert cached.metadata["fit_performance"]["statistics"]["hits"] > 0
    assert current_session() is None


def test_seed_key_compatibility():
    _, ctx, spectrum, _ = example()
    mask = spectrum.valid_mask.copy()
    key = line_seed_key(spectrum, ctx, mask)
    assert key == line_seed_key(spectrum, ctx, mask)
    assert key != line_seed_key(replace(spectrum, flux=spectrum.flux + 0.01), ctx, mask)
    assert key != line_seed_key(replace(spectrum, err=spectrum.err * 2), ctx, mask)
    mask[0] = False
    assert key != line_seed_key(spectrum, ctx, mask)
    ctx.lower = ctx.lower.copy()
    ctx.lower[0] = 1.0
    assert key != line_seed_key(spectrum, ctx, spectrum.valid_mask)


def test_workflow_warming_and_trial_isolation():
    recipe, _, spectrum, continuum = example()

    @performance_scope(workflow=True)
    def workflow(spectrum, global_config=None, *, performance=None):
        results = [
            fit_generic_complex(spectrum, continuum, recipe, n_starts=2, max_starts=2, compute_covariance=False)
            for _ in range(2)
        ]
        assert not results[0].metadata["multistart"]["warm_start"]["used"]
        assert results[1].metadata["multistart"]["warm_start"]["used"]
        explicit = fit_generic_complex(
            spectrum, continuum, recipe, n_starts=2, max_starts=2, initial_values={"line.flux": 40.0}
        )
        assert not explicit.metadata["multistart"]["warm_start"]["used"]
        single = fit_generic_complex(spectrum, continuum, recipe)
        assert not single.metadata["multistart"]["warm_start"]["used"]
        return results[-1]

    config = GlobalContinuumConfig(performance=FitPerformanceConfig(warm_starts=True))
    workflow(spectrum, config)
    workflow(spectrum, config)
    assert current_session() is None


def test_nested_override_exception_cleanup():
    @performance_scope(workflow=True)
    def workflow(global_config=None, *, performance=None):
        outer = current_session()

        @performance_scope()
        def child(*, performance=None):
            assert current_session().config == performance
            raise RuntimeError("trial")

        with pytest.raises(RuntimeError):
            child(performance=FitPerformanceConfig(cache_evaluations=False))
        assert current_session() is outer
        return SimpleNamespace(metadata={})

    workflow()
    assert current_session() is None


@pytest.mark.parametrize(
    "reason", ["starts_disagree", "oiii_parameter_at_bound", "coherent_doublet_residual", "unsuccessful_search"]
)
def test_warm_retry_archives_attempts(reason):
    solution = OptimizeResult(success=True, x=np.ones(1))
    base = dict(seed=1729, selected_start=0, starts=[{"nfev": 5}], expansion_reasons=[], search_failed=False)
    warm = {
        **base,
        "expansion_reasons": [] if reason == "unsuccessful_search" else [reason],
        "search_failed": reason == "unsuccessful_search",
    }
    with patch(
        "qsospec.fitting.multistart._solve_multistart",
        side_effect=[(solution, "vp", None, warm), (solution, "vp", None, base)],
    ) as solve:
        result = solve_multistart(None, None, None, None, None, warm_values={"x": 1.0}, n_starts=6)
    assert solve.call_args_list[0].kwargs["first_start"] == {"x": 1.0}
    assert "first_start" not in solve.call_args_list[1].kwargs
    record = result[3]
    assert record["total_nfev"] == 10 and record["total_optimizer_calls"] == 2
    assert record["warm_start"]["retry_reasons"] == [reason]
    assert [a["kind"] for a in record["warm_start"]["attempts"]] == ["warm", "cold_retry"]


def test_only_first_start_changes():
    _, context, spectrum, _ = example()
    starts = []

    def solve(ctx, wave, flux, error, initial, config):
        starts.append(initial.copy())
        return OptimizeResult(x=initial.copy(), success=True, status=1, message="ok", nfev=1), "vp", None

    config = SimpleNamespace(optimizer_method="auto")
    explicit = [dict(zip(context.names, context.initial)) for _ in range(3)]
    with patch("qsospec.fitting.global_fit._solve_once_with_fallback", solve):
        solve_multistart(
            context,
            spectrum.wave_rest,
            spectrum.flux,
            spectrum.err,
            config,
            n_starts=3,
            max_starts=3,
            start_values=explicit,
            warm_values={"line.flux": 19.0},
        )
    assert starts[0][context.index["line.flux"]] == 19.0
    assert np.array_equal(starts[1], context.initial) and np.array_equal(starts[2], context.initial)


def test_performance_and_continuum_status_persist(tmp_path):
    import qsospec
    from qsospec.io.run_store import RunStore, workflow_payload

    wave = np.linspace(3500.0, 5600.0, 200)
    spectrum = Spectrum.from_arrays(
        wave, 3 * (wave / 4000) ** -1.2, err=np.full_like(wave, 0.1), wave_frame="rest", flux_unit="relative"
    )
    config = GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,
    )
    result = qsospec.fit_global_lines(spectrum, config, complexes=[])
    # Termination status is distinct from the success flag and must survive reload.
    result.continuum.status = 3
    store = RunStore.create(str(tmp_path / "run"), configuration={"config": config})
    store.write_payload(
        workflow_payload(result, run_id=store.run_id, object_key="test", object_id="test", input_record={})
    )
    loaded = qsospec.load_model(store, "test")
    assert loaded.continuum.status == 3
    assert loaded.continuum.chi2 == result.continuum.chi2
    assert loaded.continuum.dof == result.continuum.dof
    assert loaded.metadata["fit_performance"] == result.metadata["fit_performance"]
    assert loaded.continuum.metadata["fit_performance"] == result.continuum.metadata["fit_performance"]
    assert result.metadata["fit_performance"]["statistics"]["optimizer_calls"] > 0


@pytest.mark.parametrize(
    "components,error,masked",
    [
        ([(70, 0, 320), (60, 0, 1300)], 0.03, False),
        ([(70, 0, 300), (65, 600, 1300)], 0.03, False),
        ([(65, -100, 220), (90, -800, 2300), (45, 550, 450)], 0.02, False),
        ([(0.01, 0, 300)], 0.2, False),
        ([(0, 0, 300)], 0.2, True),
    ],
)
def test_repeated_adaptive_warm_synthetic(components, error, masked):
    from test_adaptive_oiii import synthetic
    from qsospec import fit_hbeta_complex

    spectrum, continuum, config = synthetic(components, error=error)
    if masked:
        mask = spectrum.valid_mask.copy()
        mask[700:705] = False
        spectrum = replace(spectrum, mask=mask)

    @performance_scope(workflow=True)
    def workflow(global_config=None, *, performance=None):
        first = fit_hbeta_complex(spectrum, continuum, config)
        second = fit_hbeta_complex(spectrum, continuum, config)
        assert second.selected_model == first.selected_model
        assert second.warning_codes() == first.warning_codes()
        assert second.metadata["input_redshift"] == spectrum.z
        assert any(c["multistart"]["warm_start"]["used"] for c in second.metadata["candidate_selection"])
        assert not any(c["multistart"]["warm_start"]["used"] for c in first.metadata["candidate_selection"])
        return second

    workflow(GlobalContinuumConfig(performance=FitPerformanceConfig(warm_starts=True)))


def test_cache_isolation_and_singular_profile():
    from qsospec.solvers.variable_projection import _VariableProjectionProblem

    wave = np.linspace(-1.0, 1.0, 20)

    def evaluator(theta, need):
        design = np.column_stack([np.ones_like(wave), np.ones_like(wave)])
        return design, (np.zeros_like(design),) if need else None

    a = ExactEvaluationCache(evaluator, wave)
    b = ExactEvaluationCache(evaluator, wave)
    first = a(np.array([1.0]), True)
    second = b(np.array([1.0]), True)
    assert not np.shares_memory(first[0], second[0])
    problem = _VariableProjectionProblem(np.ones_like(wave), np.ones_like(wave), (np.zeros(2), np.full(2, np.inf)), a)
    assert np.all(np.isfinite(problem.jacobian(np.array([1.0]))))
