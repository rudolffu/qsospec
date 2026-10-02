"""Iron-template loading and local fitting tests for qsospec."""

import numpy as np
import pytest

import qsospec
from qsospec.jacobian import model_jacobian_dense
from qsospec.parameters import pack_line_complex_parameters
from qsospec.residuals import model_vector
from qsospec.templates import IronTemplateError, load_iron_template, prepare_iron_template
from qsospec.templates.iron import (
    evaluate_iron_basis,
    evaluate_iron_basis_with_derivative,
)


def _write_template(path, wave, flux):
    data = np.column_stack([wave, flux])
    np.savetxt(path, data)
    return path


def _iron_template_file(tmp_path):
    wave = np.linspace(4550.0, 5150.0, 301)
    flux = np.exp(-0.5 * ((wave - 4740.0) / 35.0) ** 2) + 0.6 * np.exp(-0.5 * ((wave - 5030.0) / 45.0) ** 2)
    return _write_template(tmp_path / "external_iron.txt", wave, flux)


def _hb_config(iron=None):
    return qsospec.LineComplexConfig(
        name="Hb_test",
        center=4861.33,
        window=(4700.0, 5100.0),
        components=[
            qsospec.GaussianComponent(
                name="Hb_broad",
                center=4860.0,
                amp=4.0,
                sigma=18.0,
                bounds={"amp": (0.0, None), "center": (4820.0, 4900.0), "sigma": (5.0, 80.0)},
            )
        ],
        local_continuum="linear",
        iron=iron,
    )


def test_external_template_is_area_normalized(tmp_path):
    path = _iron_template_file(tmp_path)

    template = load_iron_template("external", template_path=str(path))

    assert template.name == "external"
    assert template.coverage == (4550.0, 5150.0)
    np.testing.assert_allclose(np.trapezoid(template.flux, template.wave_rest), 1.0, rtol=1e-12)


def test_template_parse_errors_have_stable_codes(tmp_path):
    descending = _write_template(tmp_path / "bad_order.txt", [5000.0, 4990.0], [1.0, 1.0])
    nonfinite = _write_template(tmp_path / "nonfinite.txt", [4990.0, 5000.0], [1.0, np.nan])
    zero = _write_template(tmp_path / "zero.txt", [4990.0, 5000.0], [0.0, 0.0])

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external", template_path=str(descending))
    assert excinfo.value.code == "iron_template_not_monotonic"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external", template_path=str(nonfinite))
    assert excinfo.value.code == "iron_template_parse_failed"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external", template_path=str(zero))
    assert excinfo.value.code == "iron_template_zero_norm"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("not_a_template")
    assert excinfo.value.code == "unknown_iron_template"

    with pytest.raises(IronTemplateError) as excinfo:
        load_iron_template("external")
    assert excinfo.value.code == "missing_iron_template_path"


def test_prepare_template_reports_partial_and_no_overlap(tmp_path):
    template = load_iron_template("external", template_path=str(_iron_template_file(tmp_path)))
    wave_fit = np.linspace(4700.0, 5100.0, 120)

    partial = prepare_iron_template(template, wave_fit, (4500.0, 5100.0), fwhm_kms=1200.0)
    assert partial.has_overlap
    assert "iron_template_partial_coverage" in [warning.code for warning in partial.warnings]

    no_overlap = prepare_iron_template(template, wave_fit, (6000.0, 6200.0), fwhm_kms=1200.0)
    assert not no_overlap.has_overlap
    assert np.all(no_overlap.basis == 0.0)
    assert "iron_template_no_overlap" in [warning.code for warning in no_overlap.warnings]


def test_iron_amplitude_and_fwhm_jacobian_match_finite_difference(tmp_path):
    wave = np.linspace(4700.0, 5100.0, 80)
    template = load_iron_template("external", template_path=str(_iron_template_file(tmp_path)))
    basis = prepare_iron_template(template, wave, (4700.0, 5100.0), fwhm_kms=1400.0).basis
    config = _hb_config(
        iron=qsospec.IronTemplateConfig(
            template="external",
            template_path="unused.txt",
            fwhm_kms=1400.0,
            fwhm_bounds=(800.0, 2600.0),
        )
    )
    packed = pack_line_complex_parameters(
        config,
        wave,
        flux_fit=np.ones_like(wave),
        iron_basis=basis,
        iron_template=template,
    )
    theta = packed.initial.copy()
    assert packed.iron_index is not None
    assert packed.iron_fwhm_index is not None

    jac = model_jacobian_dense(theta, packed, wave)
    for index, step in [(packed.iron_index, 1.0e-6), (packed.iron_fwhm_index, 1.0)]:
        hi = theta.copy()
        lo = theta.copy()
        hi[index] += step
        lo[index] -= step
        finite_difference = (model_vector(hi, packed, wave) - model_vector(lo, packed, wave)) / (2.0 * step)

        np.testing.assert_allclose(jac[:, index], finite_difference, rtol=2e-3, atol=2e-6)


def test_fit_line_complex_recovers_external_iron_amplitude(tmp_path):
    path = _iron_template_file(tmp_path)
    wave = np.linspace(4700.0, 5100.0, 260)
    template = load_iron_template("external", template_path=str(path))
    true_fwhm = 1800.0
    prepared = prepare_iron_template(template, wave, (4700.0, 5100.0), fwhm_kms=true_fwhm)
    true_iron_amp = 1800.0
    line = 5.0 * np.exp(-0.5 * ((wave - 4862.0) / 20.0) ** 2)
    continuum = 1.2 + 0.0004 * (wave - 4900.0)
    flux = continuum + line + true_iron_amp * prepared.basis
    err = np.full_like(wave, 0.03)
    spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest", survey="desi")
    config = _hb_config(
        iron=qsospec.IronTemplateConfig(
            template="external",
            template_path=str(path),
            amp=1000.0,
            amp_bounds=(0.0, 5000.0),
            fwhm_kms=1200.0,
            fwhm_bounds=(800.0, 3500.0),
        )
    )

    result = qsospec.fit_line_complex(spectrum, config)
    table = result.to_table()
    iron_row = table[table["component_type"] == "iron"].iloc[0]

    assert result.success
    assert "iron" in result.component_models
    assert abs(result.param_values["iron.amp"] - true_iron_amp) < 80.0
    assert abs(result.param_values["iron.fwhm_kms"] - true_fwhm) < 300.0
    assert iron_row["iron_template"] == "external"
    assert abs(iron_row["iron_fwhm_kms"] - true_fwhm) < 300.0
    assert np.isfinite(iron_row["iron_flux_input"])
    assert np.isfinite(iron_row["iron_flux_cgs"])


def test_bundled_template_aliases_work_in_recipes():
    for template_name in ["bg92", "park22", "vc04"]:
        wave = np.linspace(4700.0, 5100.0, 220)
        flux = 1.0 + 5.0 * np.exp(-0.5 * ((wave - 4861.33) / 22.0) ** 2)
        err = np.full_like(wave, 0.05)
        spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest", flux_unit="relative")
        result = qsospec.fit_line_complex(spectrum, qsospec.recipes.local_hbeta(iron_template=template_name))
        assert result.success
        assert "iron.amp" in result.param_values
        assert result.metadata["iron"]["template"] in {"bg92_optical", "park22_optical", "veron04_optical"}

    wave = np.linspace(2700.0, 2900.0, 180)
    flux = 1.0 + 4.0 * np.exp(-0.5 * ((wave - 2798.75) / 18.0) ** 2)
    err = np.full_like(wave, 0.05)
    spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest", flux_unit="relative")
    result = qsospec.fit_line_complex(spectrum, qsospec.recipes.local_mgii(iron_template="vw01"))

    assert result.success
    assert result.metadata["iron"]["template"] == "vw01_uv"


def test_no_overlap_warns_and_drops_iron_parameter():
    wave = np.linspace(4700.0, 5100.0, 160)
    flux = 1.0 + 4.0 * np.exp(-0.5 * ((wave - 4861.33) / 22.0) ** 2)
    err = np.full_like(wave, 0.05)
    spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest", flux_unit="relative")

    result = qsospec.fit_line_complex(spectrum, qsospec.recipes.local_hbeta(iron_template="vw01"))

    assert result.success
    assert "iron_template_no_overlap" in result.warning_codes()
    assert "iron.amp" not in result.param_values
    assert result.metadata["iron"]["has_overlap"] is False


def test_fit_local_no_overlap_warning_does_not_crash_other_windows():
    wave = np.linspace(2700.0, 5100.0, 600)
    mgii = 3.0 * np.exp(-0.5 * ((wave - 2798.75) / 18.0) ** 2)
    hb = 4.0 * np.exp(-0.5 * ((wave - 4861.33) / 22.0) ** 2)
    flux = 1.0 + mgii + hb
    err = np.full_like(wave, 0.05)
    spectrum = qsospec.Spectrum.from_arrays(wave, flux, err=err, z=0.0, wave_frame="rest", flux_unit="relative")
    config = qsospec.LocalFitConfig(
        windows=[
            qsospec.recipes.local_hbeta(iron_template="vw01"),
            qsospec.recipes.local_mgii(iron_template="vw01"),
        ]
    )

    result = qsospec.fit_local(spectrum, config)

    assert result.success
    assert result.window_results["Hb_OIII"].success
    assert result.window_results["MgII"].success
    assert "iron_template_no_overlap" in result.warning_codes()
    assert "iron.amp" not in result.window_results["Hb_OIII"].param_values
    assert "iron.amp" in result.window_results["MgII"].param_values


def test_verner09_resource_aliases_and_native_resolution():
    canonical = load_iron_template("verner09")

    assert canonical.name == "verner09"
    assert canonical.native_fwhm_kms == 900.0
    assert canonical.coverage[0] >= 2000.0
    assert canonical.coverage[1] <= 10000.0
    assert np.all(np.diff(canonical.wave_rest) > 0)
    assert np.trapezoid(canonical.flux, canonical.wave_rest) == pytest.approx(1.0)
    for alias in ("verner_2009", "v09", "verner"):
        loaded = load_iron_template(alias)
        np.testing.assert_allclose(loaded.wave_rest, canonical.wave_rest)
        np.testing.assert_allclose(loaded.flux, canonical.flux)


def test_verner09_target_fwhm_derivative_matches_finite_difference():
    template = load_iron_template("verner09")
    wave = np.linspace(2200.0, 9800.0, 900)
    target_fwhm = 3200.0
    basis, derivative = evaluate_iron_basis_with_derivative(
        template,
        wave,
        target_fwhm,
    )
    step = 1.0
    finite_difference = (
        evaluate_iron_basis(template, wave, target_fwhm + step)
        - evaluate_iron_basis(template, wave, target_fwhm - step)
    ) / (2.0 * step)

    assert np.any(basis > 0)
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=3e-3,
        atol=2e-9,
    )
    assert np.all(np.isfinite(evaluate_iron_basis(template, wave, 900.0)))
    with pytest.raises(IronTemplateError):
        evaluate_iron_basis(template, wave, 899.0)


def test_single_verner09_global_fit_uses_one_iron_component():
    wave = np.linspace(2000.0, 10000.0, 1600)
    template = load_iron_template("verner09")
    iron = 5200.0 * evaluate_iron_basis(template, wave, 3300.0)
    power_law = 2.2 * (wave / 3000.0) ** -1.1
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        power_law + iron,
        err=np.full_like(wave, 0.002),
        wave_frame="rest",
        flux_unit="relative",
    )
    config = qsospec.GlobalContinuumConfig.with_single_iron(
        qsospec.IronTemplateConfig.verner09(
            amp=5000.0,
            fwhm_kms=3000.0,
        ),
        power_law=qsospec.PowerLawConfig(norm=2.0, slope=-1.0),
        polynomial=qsospec.PolynomialContinuumConfig(mode="off"),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            enabled=False
        ),
        continuum_windows=((2000.0, 10000.0),),
        mask_windows=(),
        clip_passes=0,
        blue_absorption_clip_enabled=False,
    )

    result = qsospec.fit_global_continuum(spectrum, config)

    assert result.success
    assert "full_iron" in result.component_models
    assert "uv_iron" not in result.component_models
    assert "optical_iron" not in result.component_models
    assert "full_iron.amp" in result.param_values
    assert "full_iron.fwhm_kms" in result.param_values
    assert result.param_values["full_iron.amp"] == pytest.approx(5200.0, rel=0.03)
    assert result.param_values["full_iron.fwhm_kms"] == pytest.approx(
        3300.0,
        rel=0.08,
    )
    assert result.metadata["iron_mode"] == "single"
    metadata = result.metadata["iron_templates"]["full_iron"]
    assert metadata["native_fwhm_kms"] == 900.0
    assert metadata["target_fwhm_kms"] == pytest.approx(
        result.param_values["full_iron.fwhm_kms"]
    )
    assert metadata["convolution_fwhm_kms"] < metadata["target_fwhm_kms"]


def test_single_verner09_string_preset_uses_safe_width_bound():
    config = qsospec.GlobalContinuumConfig.with_single_iron("v09")

    assert config.uv_iron is None
    assert config.optical_iron is None
    assert config.full_iron.template == "verner09"
    assert config.full_iron.fwhm_bounds[0] > 900.0


def test_regional_partition_and_shared_kernel_jacobian():
    from qsospec.templates.iron import regional_weights, resolve_regional_intervals
    from qsospec.fitting.global_fit import _ContinuumContext
    uv, optical = (load_iron_template(name) for name in ('vw01','park22'))
    intervals = resolve_regional_intervals(uv,optical,10000.,10000.)
    wave = np.linspace(2000.,5500.,1500)
    weights = np.array(regional_weights(wave,*intervals))
    assert np.all((weights>=0)&(weights<=1))
    np.testing.assert_allclose(weights.sum(axis=0),1.,atol=1e-15)
    spectrum = qsospec.Spectrum.from_arrays(wave,np.ones_like(wave),err=np.ones_like(wave),wave_frame='rest',flux_unit='relative')
    config = qsospec.GlobalContinuumConfig(power_law=qsospec.PowerLawConfig(mode='single'),continuum_windows=((2000.,5500.),),balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False))
    ctx = _ContinuumContext(spectrum,config)
    assert 'middle_iron.amp' in ctx.names
    assert not any(name.startswith('middle_iron.') and not name.endswith('.amp') for name in ctx.names)
    assert ctx.bridge_parent == 'optical_iron'
    nonlinear = np.array([ctx.initial[ctx.index[name]] for name in ctx.nonlinear_names])
    design, derivatives = ctx.separable_design(nonlinear,wave,True)
    index = ctx.nonlinear_names.index('optical_iron.fwhm_kms')
    plus,minus=nonlinear.copy(),nonlinear.copy();plus[index]+=1.;minus[index]-=1.
    difference=(ctx.separable_design(plus,wave,False)[0]-ctx.separable_design(minus,wave,False)[0])/2
    np.testing.assert_allclose(derivatives[index],difference,rtol=0.005,atol=2e-9)
    middle = design[:,ctx.linear_names.index('middle_iron.amp')]
    assert np.any(middle[(wave>3500)&(wave<4000)]>0)


def test_explicit_native_width_status_and_zero_kernel():
    from qsospec.templates.iron import resolve_iron_width, evaluate_iron_kernel
    verner=load_iron_template('verner09');park=load_iron_template('park22')
    assert resolve_iron_width(park,3000.)['native_fwhm_kms'] is None
    assert resolve_iron_width(park,3000.)['target_fwhm_kms'] is None
    with pytest.raises(ValueError):
        resolve_iron_width(park,3000.,'target')
    assert resolve_iron_width(verner,900.,'target')['kernel_fwhm_kms']==0.
    basis, derivative=evaluate_iron_kernel(verner,np.linspace(3300.,4500.,100),0.)
    assert np.all(np.isfinite(basis))
    np.testing.assert_array_equal(derivative,0.)


def test_soft_iron_prior_accounting_and_positive_domain():
    wave=np.linspace(2500.,5500.,350)
    spectrum=qsospec.Spectrum.from_arrays(wave,2.*(wave/3000.)**-1.,err=np.full_like(wave,.1),wave_frame='rest',flux_unit='relative')
    config=qsospec.GlobalContinuumConfig(iron_width_coupling='soft',clip_passes=0,blue_absorption_clip_enabled=False,
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False))
    fit=qsospec.fit_global_continuum(spectrum,config)
    assert fit.metadata['total_objective']==pytest.approx(fit.chi2+fit.metadata['prior_penalty'])
    assert fit.dof==int(fit.clip_mask.sum())-len(fit.param_values)
    assert fit.metadata['iron_width_prior']['basis']=='additional_kernel'
    with pytest.raises(ValueError,match='positive lower'):
        qsospec.GlobalContinuumConfig(iron_width_coupling='soft',uv_iron=qsospec.IronTemplateConfig.vw01(fwhm_bounds=(0.,10000.)))


def test_soft_width_prior_yields_to_strong_spectral_information():
    from dataclasses import replace
    from qsospec.fitting.global_fit import _ContinuumContext
    wave=np.linspace(2500.,5500.,650)
    spectrum=qsospec.Spectrum.from_arrays(wave,np.ones_like(wave),err=np.full_like(wave,.002),wave_frame='rest',flux_unit='relative')
    config=qsospec.GlobalContinuumConfig(power_law=qsospec.PowerLawConfig(norm=2.,slope=-1.,mode='single'),
        uv_iron=qsospec.IronTemplateConfig.vw01(amp=400.,fwhm_kms=1800.),
        optical_iron=qsospec.IronTemplateConfig.park22(amp=400.,fwhm_kms=6000.),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        clip_passes=0,blue_absorption_clip_enabled=False)
    context=_ContinuumContext(spectrum,config)
    truth=context.initial.copy()
    for key,value in {'power_law.norm':2.,'power_law.slope':-1.,'uv_iron.amp':400.,'optical_iron.amp':400.,'middle_iron.amp':40.}.items():
        truth[context.index[key]]=value
    spectrum=replace(spectrum,flux=context.model(truth,wave))
    result=qsospec.fit_global_continuum(spectrum,replace(config,iron_width_coupling='soft'))
    assert result.success
    assert result.param_values['uv_iron.fwhm_kms']/result.param_values['optical_iron.fwhm_kms']==pytest.approx(.3,rel=.03)
    assert result.metadata['prior_penalty']>1.


def _direct_broadened(template, fwhm_kms, velocity_step=25.0):
    """Reference implementation of the pre-optimization convolution path."""
    from qsospec.templates.iron import (
        C_KMS,
        FWHM_TO_SIGMA,
        _log_grid,
        resolve_iron_width,
    )

    resolved = resolve_iron_width(template, fwhm_kms, "legacy")
    convolution_fwhm = float(resolved["kernel_fwhm_kms"])
    native_fwhm = (
        template.native_fwhm_kms
        if resolved["requested_width_mode"] == "target"
        else 0.0
    )
    grid = _log_grid(
        float(template.wave_rest.min()), float(template.wave_rest.max()), velocity_step
    )
    sampled = np.interp(grid, template.wave_rest, template.flux, left=0.0, right=0.0)
    if convolution_fwhm == 0:
        return grid, sampled, np.zeros_like(sampled)
    sigma_pix = (convolution_fwhm / FWHM_TO_SIGMA) / float(velocity_step)
    half = max(1, int(np.ceil(4.0 * sigma_pix)))
    x = np.arange(-half, half + 1, dtype=float)
    raw_kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    raw_derivative = raw_kernel * x**2 / sigma_pix**3
    kernel_sum = raw_kernel.sum()
    kernel = raw_kernel / kernel_sum
    kernel_derivative_sigma = (
        raw_derivative * kernel_sum - raw_kernel * raw_derivative.sum()
    ) / kernel_sum**2
    chain = float(fwhm_kms) / convolution_fwhm if native_fwhm > 0 else 1.0
    kernel_derivative_fwhm = kernel_derivative_sigma * chain / (
        FWHM_TO_SIGMA * float(velocity_step)
    )
    return (
        grid,
        np.convolve(np.pad(sampled, half), kernel, mode="same")[half:-half],
        np.convolve(np.pad(sampled, half), kernel_derivative_fwhm, mode="same")[
            half:-half
        ],
    )


@pytest.mark.parametrize("template_name", ["vw01", "park22", "verner09", "bg92"])
def test_optimized_iron_convolution_matches_direct_reference(template_name):
    from qsospec.templates.iron import (
        _broaden_template_with_derivative,
        clear_iron_caches,
    )

    template = load_iron_template(template_name)
    native = template.native_fwhm_kms
    for fwhm in (0.0, 900.0, 3000.0, 12000.0):
        if native > 0 and fwhm < native:
            continue
        clear_iron_caches()
        reference_grid, reference_flux, reference_derivative = _direct_broadened(
            template, fwhm
        )
        grid, flux, derivative = _broaden_template_with_derivative(template, fwhm)
        np.testing.assert_allclose(grid, reference_grid, rtol=0.0, atol=0.0)
        scale = max(float(np.max(np.abs(reference_flux))), 1.0e-30)
        np.testing.assert_allclose(flux, reference_flux, rtol=1.0e-10, atol=1.0e-12 * scale)
        derivative_scale = max(float(np.max(np.abs(reference_derivative))), 1.0e-30)
        np.testing.assert_allclose(
            derivative,
            reference_derivative,
            rtol=1.0e-10,
            atol=1.0e-12 * derivative_scale,
        )


def test_optimized_iron_convolution_matches_direct_reference_external(tmp_path):
    from qsospec.templates.iron import (
        _broaden_template_with_derivative,
        clear_iron_caches,
    )

    path = _write_template(
        tmp_path / "external_iron.txt",
        np.linspace(2900.0, 3600.0, 501),
        np.exp(-0.5 * ((np.linspace(2900.0, 3600.0, 501) - 3200.0) / 40.0) ** 2),
    )
    template = load_iron_template("external", template_path=str(path))
    for fwhm in (0.0, 1500.0, 9000.0):
        clear_iron_caches()
        reference = _direct_broadened(template, fwhm)
        optimized = _broaden_template_with_derivative(template, fwhm)
        for expected, observed in zip(reference, optimized):
            scale = max(float(np.max(np.abs(expected))), 1.0e-30)
            np.testing.assert_allclose(
                observed, expected, rtol=1.0e-10, atol=1.0e-12 * scale
            )


def test_iron_cache_reuse_and_returned_array_immutability(monkeypatch):
    import qsospec.templates.iron as iron_module
    from qsospec.templates.iron import clear_iron_caches

    clear_iron_caches()
    calls = []
    original = iron_module._linear_convolve

    def counting(padded, kernel):
        calls.append(1)
        return original(padded, kernel)

    monkeypatch.setattr(iron_module, "_linear_convolve", counting)
    template = load_iron_template("park22")
    wave = np.linspace(4000.0, 5500.0, 500)
    first = evaluate_iron_basis(template, wave, 3000.0)
    assert len(calls) == 1
    first += 1234.0
    second = evaluate_iron_basis(template, wave, 3000.0)
    assert len(calls) == 1
    assert not np.allclose(first, second)
    expected = _direct_broadened(template, 3000.0)
    from qsospec.templates.iron import _apply_coverage_taper

    reference = _apply_coverage_taper(
        template, wave, np.interp(wave, expected[0], expected[1], left=0.0, right=0.0)
    )
    np.testing.assert_allclose(second, reference, rtol=1.0e-10, atol=1.0e-12)


def test_flux_only_iron_evaluation_skips_derivative_convolutions(monkeypatch):
    import qsospec.templates.iron as iron_module
    from qsospec.templates.iron import clear_iron_caches

    template = load_iron_template("vw01")
    wave = np.linspace(2200.0, 3100.0, 400)
    clear_iron_caches()
    derivative_calls = []
    original = iron_module._broadened_derivative

    def counting(*args, **kwargs):
        derivative_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(iron_module, "_broadened_derivative", counting)
    evaluate_iron_basis(template, wave, 2600.0)
    assert derivative_calls == []
    evaluate_iron_basis_with_derivative(template, wave, 2600.0)
    assert len(derivative_calls) == 1


def test_regional_bridge_value_only_design_convolves_once_per_template(monkeypatch):
    import qsospec.templates.iron as iron_module
    from qsospec.fitting.global_fit import _ContinuumContext
    from qsospec.templates.iron import clear_iron_caches

    wave = np.linspace(3300.0, 5500.0, 400)
    spectrum = qsospec.Spectrum.from_arrays(
        wave,
        2.0 * (wave / 4000.0) ** -1.0,
        err=np.full_like(wave, 0.05),
        wave_frame="rest",
        flux_unit="relative",
    )
    clear_iron_caches()
    calls = []
    original = iron_module._linear_convolve

    def counting(padded, kernel):
        calls.append(1)
        return original(padded, kernel)

    monkeypatch.setattr(iron_module, "_linear_convolve", counting)
    context = _ContinuumContext(spectrum, qsospec.GlobalContinuumConfig())
    nonlinear = np.array(
        [context.initial[context.index[name]] for name in context.nonlinear_names]
    )
    context.separable_design(nonlinear, wave, False)
    # One convolution for uv_iron, optical_iron, and the regional middle
    # template; the legacy outer taper reuses the same broadened arrays and
    # the repeated value-only design is served from the cache.
    assert len(calls) == 3


def test_iron_basis_is_zero_beyond_template_coverage():
    template = load_iron_template("park22")
    coverage = template.coverage
    wave = np.linspace(coverage[0] - 40.0, coverage[1] + 40.0, 500)
    basis = evaluate_iron_basis(template, wave, 3000.0)
    assert np.all(basis[(wave < coverage[0]) | (wave > coverage[1])] == 0.0)
    assert np.all(basis >= 0.0)
    assert np.any(basis > 0.0)
