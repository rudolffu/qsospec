"""Capture deterministic synthetic science outputs before/after API cleanup.

Run against either source tree with PYTHONPATH=/path/to/source/src. Compare
the two NPZ products with --compare; all coordinates are physical or native
point estimates, independent of draw-key spelling and runtime metadata.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import qsospec
from qsospec.fitting.global_fit import _ContinuumContext, _gaussian_area_profile
from qsospec.templates import load_balmer_template, load_iron_template
from qsospec.templates.iron import evaluate_iron_basis
from qsospec.workflows.host.agn_templates import build_host_agn_template_bundle
from qsospec.workflows.host.config import HostAgnPseudoContinuumConfig
from qsospec.workflows.host.io import SpectrumData
from qsospec.workflows.host.ppxf_host import prepare_spectrum_for_host_decomp, run_ppxf_host_fit
from qsospec.workflows.host.templates import PPXFTemplateLibrary


def capture():
    output = {}

    def record(name, result):
        output[name + ".continuum_model"] = result.continuum.model
        output[name + ".continuum_parameters"] = np.array(list(result.continuum.param_values.values()))
        for recipe, fit in result.line_complexes.items():
            output[name + "." + recipe + ".model"] = fit.model
            output[name + "." + recipe + ".parameters"] = np.array(list(fit.param_values.values()))
            for metric, value in fit.metrics.items():
                output[name + "." + recipe + ".metric." + metric] = np.array(value)
        for sample, value in result.metadata["continuum_samples"].items():
            output[name + ".sample." + sample] = np.array(value)
        for row in result.monte_carlo.get("draws", []):
            # Native per-recipe parameter draws avoid the renamed namespace.
            for recipe, parameters in row["parameters"].items():
                output[name + f".trial{row['trial_id']}." + recipe] = np.array(list(parameters.values()))

    wave = np.linspace(2600.0, 5500.0, 650)
    spectrum = qsospec.Spectrum.from_arrays(
        wave, np.ones_like(wave), err=np.full_like(wave, 0.03), wave_frame="rest", flux_unit="relative"
    )
    cfg = qsospec.GlobalContinuumConfig(
        power_law=qsospec.PowerLawConfig(norm=2.0, slope=-1.0, mode="single"),
        uv_iron=qsospec.IronTemplateConfig.vw01(amp=80.0),
        optical_iron=qsospec.IronTemplateConfig.park22(amp=70.0),
        regional_iron=qsospec.RegionalIronConfig(amp=35.0),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(enabled=False),
        continuum_windows=((2600.0, 5500.0),),
        mask_windows=(),
        clip_passes=0,
        blue_absorption_clip_enabled=False,
    )
    context = _ContinuumContext(spectrum, cfg)
    theta = context.initial.copy()
    for name, value in {
        "power_law.norm": 2.0,
        "power_law.slope": -1.0,
        "uv_iron.amp": 80.0,
        "optical_iron.amp": 70.0,
        "middle_iron.amp": 35.0,
    }.items():
        theta[context.index[name]] = value
    output["regional.rendered"] = context.model(theta, wave)
    spectrum = replace(spectrum, flux=output["regional.rendered"])
    record("regional", qsospec.fit_global_lines(spectrum, cfg, complexes=()))
    for template in ("vw01", "park22", "verner09"):
        output["iron." + template] = evaluate_iron_basis(load_iron_template(template), wave, 3200.0)

    wave = np.linspace(3300.0, 5150.0, 650)
    balmer = 18.0 * qsospec.evaluate_balmer_pseudocontinuum(
        load_balmer_template(provenance="sh95_k13full_ext"), wave, 3400.0, 0.0
    )
    flux = (
        2.0
        + balmer
        + _gaussian_area_profile(wave, 22.0, 4341.68, 3300.0)
        + _gaussian_area_profile(wave, 60.0, 4862.68, 3400.0)
    )
    flux += _gaussian_area_profile(wave, 25.0, 5008.24, 350.0) + _gaussian_area_profile(
        wave, 25.0 / 2.98, 4960.30, 350.0
    )
    spectrum = qsospec.Spectrum.from_arrays(
        wave, flux, err=np.full_like(wave, 0.02), wave_frame="rest", flux_unit="relative"
    )
    cfg = qsospec.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        power_law=qsospec.PowerLawConfig(norm=2.0, slope=0.0, mode="single"),
        balmer_pseudocontinuum=qsospec.BalmerPseudoContinuumConfig(
            amplitude=18.0, fwhm_kms=3400.0, sync_with_hbeta="never", sync_with_hgamma="soft"
        ),
        continuum_windows=((3300.0, 4260.0),),
        mask_windows=(),
        clip_passes=0,
        blue_absorption_clip_enabled=False,
    )
    result = qsospec.fit_global_lines(
        spectrum,
        cfg,
        complexes=("oii_nev_neiii_hgamma", "hbeta_oiii"),
        hbeta_config=qsospec.HbetaComplexConfig(fit_oiii_wings=False),
        uncertainty_config=qsospec.UncertaintyConfig(monte_carlo_trials=2, random_seed=1729),
    )
    record("soft_hgamma_hbeta", result)
    output["soft_hgamma.delta_dex"] = np.array(result.continuum.metadata["hgamma_delta_dex"])

    wave = np.linspace(3600.0, 7000.0, 700)
    bundle = build_host_agn_template_bundle(wave, selected_fwhm_kms=3000.0, config=HostAgnPseudoContinuumConfig())
    for component in bundle.components:
        output["host_basis." + component.name] = component.values
    template_wave = np.linspace(3500.0, 7100.0, 900)
    stellar = (
        1.0
        - 0.2 * np.exp(-0.5 * ((template_wave - 5175.0) / 12.0) ** 2)
        - 0.15 * np.exp(-0.5 * ((template_wave - 4300.0) / 15.0) ** 2)
    )
    templates = PPXFTemplateLibrary(
        flux=stellar[:, None],
        wave=template_wave,
        log_wave=np.log(template_wave),
        family="synthetic",
        source_path="synthetic.npz",
        wavelength_coverage=(3500.0, 7100.0),
    )
    flux = 0.7 * np.interp(wave, template_wave, stellar) + 0.3
    prep = prepare_spectrum_for_host_decomp(
        SpectrumData(wave_obs=wave, flux=flux, ivar=np.full_like(wave, 2500.0), redshift=0.0)
    )
    host = run_ppxf_host_fit(prep, templates, agn_powerlaw_slopes=(0.0,))
    output["host.model"] = host.host_model
    output["host.fraction"] = host.host_model / host.total_model
    preset = (
        qsospec.recipes.extended_quasar()
        if hasattr(qsospec.recipes, "extended_quasar")
        else qsospec.recipes.nir_complete()
    )
    output["preset.components_json"] = np.array(json.dumps([asdict(recipe) for recipe in preset], sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        before, after = (np.load(p) for p in args.compare)
        assert set(before.files) == set(after.files)
        maximum = 0.0
        for name in before.files:
            if before[name].dtype.kind in "US":
                np.testing.assert_array_equal(before[name], after[name])
            else:
                np.testing.assert_allclose(
                    before[name], after[name], rtol=1.0e-10, atol=1.0e-10, equal_nan=True, err_msg=name
                )
                delta = np.abs(before[name] - after[name])
                finite = delta[np.isfinite(delta)]
                if finite.size:
                    maximum = max(maximum, float(finite.max()))
        print(
            json.dumps(
                {
                    "status": "passed",
                    "products_compared": len(before.files),
                    "rtol": 1.0e-10,
                    "atol": 1.0e-10,
                    "maximum_absolute_difference": maximum,
                }
            )
        )
    else:
        if not args.output:
            parser.error("--output or --compare is required")
        np.savez_compressed(Path(args.output), **capture())
        print("Saved", args.output)
