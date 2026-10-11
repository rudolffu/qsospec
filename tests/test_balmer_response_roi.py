"""Exact response pruning must preserve the likelihood and joint covariance."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

import qsospec
from qsospec.fitting import balmer_local as native
from qsospec.fitting.complexes import GenericComplexContext
from qsospec.fitting.global_fit import _gaussian_area_profile
from qsospec.solvers.variable_projection import _VariableProjectionProblem


def _long_grid(line="hbeta", *, dense=False):
    wave = np.linspace(3200., 9000., 601)
    input_wave = np.linspace(3100., 9100., 1201)
    z = .07358
    # Deliberately unsorted duplicate column entries and signed coefficients.
    # They must not be normalized, summed or reordered during sparse pruning.
    indices, values, indptr = [], [], [0]
    for row, wavelength in enumerate(wave):
        column = int(np.searchsorted(input_wave, wavelength))
        indices.extend([column+1, column-1, column, column])
        values.extend([.07, -.01, .60, .34])
        if row in (174, 348):
            # A distant native input affects a fitted row despite lying
            # outside either Balmer window. Wavelength cropping loses it.
            indices.extend([0, len(input_wave)-1])
            values.extend([.003, -.001])
        indptr.append(len(indices))
    matrix = sparse.csr_matrix((values, indices, indptr),
                              shape=(len(wave), len(input_wave)))
    if dense:
        matrix = matrix.toarray()
    valid = np.ones(len(wave), bool)
    valid[190:193] = False
    spectrum = qsospec.Spectrum.from_arrays(wave, np.ones(len(wave)),
        err=np.full(len(wave), .02), mask=valid, z=z,
        wave_frame="rest", flux_unit="relative")
    transmission_in = np.exp(-.3*(input_wave/5000.)**-1.2)
    transmission_out = np.exp(-.3*(wave/5000.)**-1.2)
    # Reversed band-row ordering tests exact output-index mapping. An overlap
    # band and foreground conjugation are both active on every fitted row.
    order = np.arange(len(wave))[::-1]
    operators = [qsospec.BandResolutionOperator("forward", input_wave*(1+z),
        np.arange(len(wave)), matrix, output_weights=.25,
        input_factor=transmission_in, output_factor=1/transmission_out),
        qsospec.BandResolutionOperator("reverse", input_wave*(1+z),
        order, matrix[order, :], output_weights=.75,
        input_factor=transmission_in, output_factor=1/transmission_out[order])]
    config = qsospec.BalmerLocalConfig(line=line)
    return spectrum, operators, config


def _context(spectrum, operators, config):
    mask, active_sii, _ = native._coverage(spectrum, config)
    return native._ResponseContext(native._recipe(config, active_sii),
                                   150., spectrum, mask, operators)


def _original_projection(context, evaluate):
    """Independent original full-domain response, before the ROI path."""
    output = None
    for operator in context.operators:
        values = evaluate(operator.input_wave_obs/(1+context.spectrum.z))
        shape = (len(operator.input_factor),)+(1,)*(values.ndim-1)
        projected = np.asarray(operator.matrix.dot(values*operator.input_factor.reshape(shape)))
        shape = (len(operator.output_factor),)+(1,)*(values.ndim-1)
        projected *= (operator.output_factor*operator.output_weights).reshape(shape)
        if output is None:
            output = np.zeros((len(context.spectrum.flux),)+values.shape[1:])
        output[operator.output_indices] += projected
    output[context.output_coverage == 0] = np.nan
    return output


@pytest.mark.parametrize("line,count,family", [
    (line, count, family) for line in ("hbeta", "halpha")
    for count in (0, 1) for family in ("core", "shared_outflow")])
@pytest.mark.parametrize("storage", ["csr64", "csr32", "dense"])
def test_exact_design_derivatives_and_components_preserve_signed_response(line, count, family, storage):
    dense = storage == "dense"
    spectrum, operators, config = _long_grid(line, dense=dense)
    if storage == "csr32":
        operators = [replace(operator, matrix=operator.matrix.astype(np.float32),
                             output_weights=operator.output_weights.astype(np.float32))
                     for operator in operators]
    config = replace(config, broad_count=count, nuisance_family=family)
    context = _context(spectrum, operators, config)
    _, _, nonlinear, _ = context.separable_initial_and_bounds()
    nonlinear = nonlinear.copy()
    nonlinear[0] = 57.
    theta = context.initial.copy()
    theta[context.index["continuum.constant"]] = 2.
    theta[context.index["continuum.slope"]] = .003
    theta[context.index["narrow.velocity_kms"]] = 57.
    fit_wave = spectrum.wave_rest[context.fit_indices]

    def design_evaluator(wave):
        design, derivatives = GenericComplexContext.separable_design(context, nonlinear, wave, True)
        return np.concatenate((design, *derivatives), axis=1)
    expected = _original_projection(context, design_evaluator)[context.fit_indices]
    design, derivatives = context.separable_design(nonlinear, fit_wave, True)
    actual = np.concatenate((design, *derivatives), axis=1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(context.separable_design(nonlinear, fit_wave, False)[0], design)

    names = list(GenericComplexContext.components(context, theta, fit_wave))
    def component_evaluator(wave):
        components = GenericComplexContext.components(context, theta, wave)
        return np.column_stack([components[name] for name in names])
    expected_components = _original_projection(context, component_evaluator)
    actual_components = context.components(theta, fit_wave)
    for index, name in enumerate(names):
        np.testing.assert_array_equal(actual_components[name], expected_components[context.fit_indices, index])
        # Final QA still uses the original full-domain response, including
        # rows far outside the fitting window.
        np.testing.assert_array_equal(context.components(theta, spectrum.wave_rest)[name], expected_components[:, index])
    cards = context.evaluation_card["bands"]
    assert context.evaluation_card["sparse_pruning_applied"] is not dense
    if dense:
        assert all(card["evaluated_input_columns"] == card["original_input_columns"] for card in cards)
        return
    assert all(card["evaluated_input_columns"] < card["original_input_columns"]/3 for card in cards)
    # At least one distant column is intentionally retained; it supplies a
    # nonzero affine-continuum contribution and cannot be discarded.
    assert any(np.any(wave < config.window[0]-1000) for wave, *_ in context.fit_operators)


def test_operator_without_fitting_rows_is_not_evaluated():
    spectrum, operators, config = _long_grid()
    context = _context(spectrum, operators, config)
    extra = qsospec.BandResolutionOperator("unrelated", np.array([1000., 1001.]),
        np.array([0, 1]), sparse.eye(2), output_weights=0.)
    context = _context(spectrum, [*operators, extra], config)
    _, _, nonlinear, _ = context.separable_initial_and_bounds()
    calls = []
    def evaluate(wave):
        calls.append(wave.copy())
        return GenericComplexContext.separable_design(context, nonlinear, wave, False)[0]
    actual = context._apply(evaluate, fit_only=True)
    assert len(calls) == 2
    assert len(actual) == len(context.fit_indices)
    assert context.evaluation_card["bands"][-1]["evaluated_input_columns"] == 0


def test_mixed_dense_sparse_operators_keep_original_dense_blas_path():
    spectrum, operators, config = _long_grid()
    operators[1] = replace(operators[1], matrix=operators[1].matrix.toarray())
    context = _context(spectrum, operators, config)
    assert context.evaluation_card["sparse_pruning_applied"] is False
    _, _, nonlinear, _ = context.separable_initial_and_bounds()
    def evaluate(wave):
        return GenericComplexContext.separable_design(context, nonlinear, wave, False)[0]
    np.testing.assert_array_equal(context._apply(evaluate, fit_only=True),
                                  _original_projection(context, evaluate)[context.fit_indices])


@pytest.mark.parametrize("line", ["hbeta", "halpha"])
def test_bounded_linear_solution_residual_and_reduced_jacobian_are_bit_identical(line):
    spectrum, operators, config = _long_grid(line)
    context = _context(spectrum, operators, replace(config, nuisance_family="shared_outflow"))
    _, bounds, nonlinear, _ = context.separable_initial_and_bounds()
    fit_wave = spectrum.wave_rest[context.fit_indices]
    def full_evaluator(theta, derivatives):
        def evaluate(wave):
            design, partials = GenericComplexContext.separable_design(context, theta, wave, derivatives)
            return np.concatenate((design, *partials), axis=1) if derivatives else design
        projected = _original_projection(context, evaluate)[context.fit_indices]
        nlinear = len(context.linear_names)
        partials = tuple(projected[:, nlinear*(i+1):nlinear*(i+2)]
                         for i in range(len(context.nonlinear_names))) if derivatives else None
        return projected[:, :nlinear], partials
    original = _VariableProjectionProblem(spectrum.flux[context.fit_indices],
        spectrum.err[context.fit_indices], bounds, full_evaluator)
    pruned = _VariableProjectionProblem(spectrum.flux[context.fit_indices],
        spectrum.err[context.fit_indices], bounds,
        lambda theta, derivatives: context.separable_design(theta, fit_wave, derivatives))
    for delta in (0., 2., -3.):
        theta = nonlinear.copy()
        theta[0] += delta
        np.testing.assert_array_equal(pruned.residual(theta), original.residual(theta))
        np.testing.assert_array_equal(pruned.jacobian(theta), original.jacobian(theta))
        np.testing.assert_array_equal(pruned._state.linear, original._state.linear)
        np.testing.assert_array_equal(pruned._state.linear_active_mask, original._state.linear_active_mask)


def test_joint_fit_parameters_covariance_and_multistarts_equal_original_response(monkeypatch):
    wave = np.linspace(4100., 7500., 2301)
    center = qsospec.lines.get("halpha").vacuum_wavelength
    intrinsic = 2.+.0003*(wave-center)
    for name, flux, width in (("halpha", 20., 320.), ("halpha", 80., 2400.),
        ("nii_6585", 15., 320.), ("nii_6550", 15./2.96, 320.),
        ("sii_6718", 8., 320.), ("sii_6733", 6., 320.)):
        intrinsic += _gaussian_area_profile(wave, flux, qsospec.lines.get(name).vacuum_wavelength, width)
    matrix = sparse.diags([-.02*np.ones(len(wave)-1), 1.04*np.ones(len(wave)),
                            -.02*np.ones(len(wave)-1)], [-1, 0, 1], format="csr")
    observed = matrix.dot(intrinsic)+np.random.default_rng(91).normal(0., .01, len(wave))
    spectrum = qsospec.Spectrum.from_arrays(wave, observed, err=np.full(len(wave), .01),
        z=.07358, wave_frame="rest", flux_unit="relative")
    operator = qsospec.BandResolutionOperator("native", spectrum.wave_obs,
        np.arange(len(wave)), matrix)
    config = qsospec.BalmerLocalConfig(line="halpha", broad_count=1,
        broad_width_starts_kms=(1200., 3000.), broad_velocity_starts_kms=(0.,))
    _, actual = qsospec.fit_balmer_local(spectrum, config, instrumental_response=[operator])
    original_context = native._ResponseContext
    class FullDomainContext(original_context):
        def _apply(self, evaluate, *, fit_only=False):
            result = _original_projection(self, evaluate)
            return result[self.fit_indices] if fit_only else result
    monkeypatch.setattr(native, "_ResponseContext", FullDomainContext)
    _, expected = qsospec.fit_balmer_local(spectrum, config, instrumental_response=[operator])
    assert actual.success and expected.success
    assert actual.param_values == expected.param_values
    assert actual.param_errors == expected.param_errors
    assert actual.chi2 == expected.chi2 and actual.bic == expected.bic
    np.testing.assert_array_equal(actual.covariance, expected.covariance)
    np.testing.assert_array_equal(actual.model, expected.model)
    assert actual.metadata["deterministic_multistarts"] == expected.metadata["deterministic_multistarts"]
