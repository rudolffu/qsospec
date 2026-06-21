"""Single-object and resumable process-parallel qsospec execution."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import numpy as np

from .host.io import SpectrumData

from ..complex_recipes import ComplexRecipe
from ..config import (
    GalacticExtinctionConfig,
    GlobalContinuumConfig,
    HalphaComplexConfig,
    HbetaComplexConfig,
    LyaNVComplexConfig,
    MgIIComplexConfig,
    UncertaintyConfig,
)
from ..extinction import (
    correct_spectrum_data,
    prepare_spectrum,
    preflight_galactic_extinction,
)
from ..fitting.global_fit import fit_global_lines
from ..io.products import GlobalQAPlotConfig, write_global_line_products
from ..global_result import WorkflowResult
from .host_workflow import (
    _host_decomp_decision,
    _host_subtracted_spectrum,
    _spectrum_from_spectrum_data,
)
from ..io.readers import (
    SpectrumInput,
    discover_fits_inputs,
    read_input_manifest,
    read_spectrum,
    scan_parquet_spectra,
)
from ..io.run_store import RunStore, finalize_run, workflow_payload
from ..spectrum import Spectrum
from ..warnings import FitWarning


@dataclass
class BatchResult:
    """Summary of one batch invocation."""

    run_directory: str
    run_id: str
    n_submitted: int
    n_completed: int
    n_failed: int
    n_skipped: int
    n_workers: int
    datasets: Dict[str, str]


@dataclass
class _Task:
    descriptor: SpectrumInput
    spectrum_data: Optional[SpectrumData]
    run_directory: str
    fit_options: Dict[str, Any]
    legacy_output: bool = False


def _auto_workers(number_of_objects: Optional[int]) -> int:
    cpu_count = os.cpu_count() or 1
    available = max(cpu_count - 1, 1)
    if number_of_objects is not None:
        available = min(available, max(number_of_objects, 1))
    return min(available, 8)


def _worker_initializer() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _process_pool_available() -> bool:
    try:
        import multiprocessing.synchronize  # noqa: F401

        os.sysconf("SC_SEM_NSEMS_MAX")
    except (ImportError, NotImplementedError, OSError, PermissionError, ValueError):
        return False
    return True


def _seed_for(run_id: str, object_key: str, base_seed: Optional[int]) -> int:
    payload = f"{run_id}|{object_key}|{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _fit_spectrum_data(
    spectrum_data: SpectrumData,
    *,
    descriptor: SpectrumInput,
    run_host_decomp: bool,
    template_root: str,
    template_file: str,
    host_fit_range,
    host_config,
    galactic_extinction_config,
    global_config,
    hbeta_config,
    mgii_config,
    halpha_config,
    lya_nv_config,
    uncertainty_config,
    complexes,
):
    spectrum_data = correct_spectrum_data(
        spectrum_data, galactic_extinction_config
    )
    source = (
        f"{descriptor.source}:row_index={descriptor.row_index}"
        if descriptor.row_index is not None else descriptor.source
    )
    host_decomp_enabled, host_skip_reason = _host_decomp_decision(
        run_host_decomp, spectrum_data.redshift
    )
    if host_decomp_enabled:
        (
            total_spectrum,
            fit_spectrum,
            host_fit,
            host_sed,
            host_on_grid,
            _,
            host_warnings,
        ) = _host_subtracted_spectrum(
            spectrum_data,
            redshift=float(spectrum_data.redshift),
            template_root=template_root,
            template_file=template_file,
            fit_range=host_fit_range,
            host_config=host_config,
            source=source,
        )
    else:
        total_spectrum = _spectrum_from_spectrum_data(
            spectrum_data, source=source
        )
        fit_spectrum = total_spectrum
        host_fit = None
        host_sed = None
        host_on_grid = None
        host_warnings = []
    result = fit_global_lines(
        fit_spectrum,
        global_config,
        hbeta_config,
        mgii_config,
        halpha_config,
        uncertainty_config,
        lya_nv_config=lya_nv_config,
        host_model_on_grid=host_on_grid,
        complexes=complexes,
    )
    result.host_decomp_enabled = host_decomp_enabled
    result.total_spectrum = total_spectrum
    result.host_fit = host_fit
    result.host_sed = host_sed
    result.host_model_on_quasar_grid = host_on_grid
    result.host_fit_mask = (
        np.asarray(host_fit.preprocessed.fit_mask, dtype=bool).copy()
        if host_fit is not None else None
    )
    result.host_emission_mask = (
        np.asarray(host_fit.preprocessed.emission_mask, dtype=bool).copy()
        if host_fit is not None else None
    )
    result.host_warnings = [str(item) for item in host_warnings]
    object_id = (
        descriptor.object_id
        or spectrum_data.object_id
        or spectrum_data.targetid
        or Path(descriptor.source).stem
    )
    result.metadata.update(
        {
            "input_path": descriptor.source,
            "row_index": descriptor.row_index,
            "object_id": str(object_id),
            "targetid": spectrum_data.targetid,
            "ra": spectrum_data.ra,
            "dec": spectrum_data.dec,
            "redshift": fit_spectrum.z,
            "fit_kind": "global",
            "host_decomp_requested": bool(run_host_decomp),
            "host_decomp_enabled": host_decomp_enabled,
            "host_decomp_skip_reason": host_skip_reason,
            "host_model_source": (
                "template_weighted_sed_on_quasar_grid"
                if host_decomp_enabled else None
            ),
            "host_fit_range": list(host_fit_range),
            "host_mask_provenance": (
                "exact" if host_decomp_enabled else "unavailable"
            ),
            "host_ppxf_status": (
                host_fit.status if host_fit is not None else None
            ),
            "host_ppxf_reduced_chi2": (
                float(host_fit.reduced_chi2)
                if host_fit is not None else None
            ),
            "host_template_file": (
                host_fit.templates.source_path
                if host_fit is not None else None
            ),
            "host_template_wavelength_coverage": (
                list(host_fit.templates.wavelength_coverage)
                if host_fit is not None else None
            ),
            "galactic_extinction": dict(
                spectrum_data.metadata.get("galactic_extinction", {})
            ),
        }
    )
    if run_host_decomp and not host_decomp_enabled:
        result.warnings.append(
            FitWarning(
                code="host_decomp_skipped_redshift",
                message="Host decomposition was requested but skipped by the redshift gate.",
                severity="info",
                context={
                    "redshift": spectrum_data.redshift,
                    "threshold": 1.2,
                    "reason": host_skip_reason,
                },
            )
        )
    return result, str(object_id)


def _failure_payload(
    store: RunStore,
    descriptor: SpectrumInput,
    exception: BaseException,
) -> Dict[str, list[dict[str, Any]]]:
    return {
        "inputs": [
            {
                "run_id": store.run_id,
                "object_key": descriptor.object_key,
                "object_id": descriptor.object_id,
                "source": descriptor.source,
                "row_index": descriptor.row_index,
                "reader": descriptor.reader,
                "redshift": descriptor.redshift,
                "metadata": [
                    {
                        "key": str(key),
                        "value": json.dumps(value, sort_keys=True, default=repr),
                    }
                    for key, value in sorted(descriptor.metadata.items())
                ],
            }
        ],
        "failures": [
            {
                "run_id": store.run_id,
                "object_key": descriptor.object_key,
                "object_id": descriptor.object_id,
                "source": descriptor.source,
                "row_index": descriptor.row_index,
                "exception_type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
                "failed_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "metadata": [],
            }
        ]
    }


def _run_task(task: _Task) -> Dict[str, Any]:
    _worker_initializer()
    store = RunStore.open(task.run_directory)
    descriptor = task.descriptor
    try:
        spectrum_data = task.spectrum_data or read_spectrum(
            descriptor.source,
            row_index=descriptor.row_index,
            redshift=descriptor.redshift,
            object_id=descriptor.object_id,
            reader=descriptor.reader,
        )
        options = dict(task.fit_options)
        uncertainty = options["uncertainty_config"]
        options["uncertainty_config"] = UncertaintyConfig(
            covariance=uncertainty.covariance,
            monte_carlo_trials=uncertainty.monte_carlo_trials,
            random_seed=_seed_for(
                store.run_id,
                descriptor.object_key,
                uncertainty.random_seed,
            ),
            refit_host_in_mc=uncertainty.refit_host_in_mc,
        )
        result, object_id = _fit_spectrum_data(
            spectrum_data,
            descriptor=descriptor,
            **options,
        )
        payload = workflow_payload(
            result,
            run_id=store.run_id,
            object_key=descriptor.object_key,
            object_id=object_id,
            input_record={
                "source": descriptor.source,
                "row_index": descriptor.row_index,
                "reader": descriptor.reader,
                "metadata": dict(descriptor.metadata),
            },
        )
        staging = store.stage_payload(payload)
        legacy_files = {}
        if task.legacy_output:
            legacy_files = write_global_line_products(
                result,
                str(store.path / "legacy" / object_id),
            )
        return {
            "success": True,
            "object_key": descriptor.object_key,
            "object_id": object_id,
            "staging": str(staging),
            "legacy_files": legacy_files,
        }
    except Exception as exception:
        staging = store.stage_payload(
            _failure_payload(store, descriptor, exception)
        )
        return {
            "success": False,
            "object_key": descriptor.object_key,
            "object_id": descriptor.object_id,
            "staging": str(staging),
            "error": str(exception),
        }


def _run_task_group(tasks: Sequence[_Task]) -> list[Dict[str, Any]]:
    """Run one Parquet microbatch or one FITS task inside a worker."""

    return [_run_task(task) for task in tasks]


def _configuration(
    *,
    run_host_decomp,
    template_root,
    template_file,
    host_fit_range,
    host_config,
    galactic_extinction_config,
    global_config,
    hbeta_config,
    mgii_config,
    halpha_config,
    lya_nv_config,
    uncertainty_config,
    complexes,
) -> Dict[str, Any]:
    return {
        "run_host_decomp": bool(run_host_decomp),
        "template_root": str(template_root),
        "template_file": str(template_file),
        "host_fit_range": tuple(host_fit_range),
        "host_config": host_config,
        "galactic_extinction_config": asdict(galactic_extinction_config),
        "global_config": (
            asdict(global_config)
            if global_config is not None
            else {"preset": "automatic_lya_safe"}
        ),
        "hbeta_config": asdict(hbeta_config),
        "mgii_config": asdict(mgii_config),
        "halpha_config": asdict(halpha_config),
        "lya_nv_config": asdict(lya_nv_config),
        "uncertainty_config": asdict(uncertainty_config),
        "complexes": [
            asdict(item) if isinstance(item, ComplexRecipe) else str(item)
            for item in complexes
        ] if complexes is not None else None,
    }


def _configuration_overrides(configuration: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _configuration(
        run_host_decomp=False,
        template_root="~/tools/ppxf_data",
        template_file="spectra_emiles_9.0.npz",
        host_fit_range=(3600.0, 7000.0),
        host_config=None,
        galactic_extinction_config=GalacticExtinctionConfig(),
        global_config=None,
        hbeta_config=HbetaComplexConfig(),
        mgii_config=MgIIComplexConfig(),
        halpha_config=HalphaComplexConfig(),
        lya_nv_config=LyaNVComplexConfig(),
        uncertainty_config=UncertaintyConfig(),
        complexes=None,
    )

    def diff(value, default):
        if isinstance(value, dict) and isinstance(default, dict):
            changed = {
                key: diff(item, default.get(key))
                for key, item in value.items()
                if key not in default or item != default.get(key)
            }
            return {key: item for key, item in changed.items() if item != {}}
        return value

    overrides = {}
    for key, value in configuration.items():
        default_value = defaults.get(key)
        if (
            key == "global_config"
            and isinstance(value, dict)
            and "preset" not in value
        ):
            default_value = asdict(GlobalContinuumConfig())
        if key not in defaults or value != default_value:
            changed = diff(value, default_value)
            if changed != {}:
                overrides[key] = changed
    return overrides


def _fit_options(
    *,
    run_host_decomp,
    template_root,
    template_file,
    host_fit_range,
    host_config,
    galactic_extinction_config,
    global_config,
    hbeta_config,
    mgii_config,
    halpha_config,
    lya_nv_config,
    uncertainty_config,
    complexes,
) -> Dict[str, Any]:
    return {
        "run_host_decomp": bool(run_host_decomp),
        "template_root": template_root,
        "template_file": template_file,
        "host_fit_range": tuple(host_fit_range),
        "host_config": host_config,
        "galactic_extinction_config": galactic_extinction_config,
        "global_config": global_config,
        "hbeta_config": hbeta_config,
        "mgii_config": mgii_config,
        "halpha_config": halpha_config,
        "lya_nv_config": lya_nv_config,
        "uncertainty_config": uncertainty_config,
        "complexes": complexes,
    }


def fit_object_to_store(
    input_data: Union[str, SpectrumInput, SpectrumData, Spectrum],
    run_directory: str,
    *,
    row_index: Optional[int] = None,
    redshift: Optional[float] = None,
    object_id: Optional[str] = None,
    reader: str = "auto",
    flux_unit: Optional[str] = None,
    flux_scale: Optional[float] = None,
    run_host_decomp: bool = False,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range=(3600.0, 7000.0),
    host_config=None,
    galactic_extinction_config: Optional[
        GalacticExtinctionConfig
    ] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    lya_nv_config: Optional[LyaNVComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    complexes: Optional[Sequence[Union[str, ComplexRecipe]]] = None,
    run_id: Optional[str] = None,
    resume: bool = True,
    write_qa: bool = True,
    qa_plot_config: Optional[GlobalQAPlotConfig] = None,
    write_legacy_products: bool = False,
) -> WorkflowResult:
    """Fit one object into the same run bundle used for batch fitting."""

    hbeta_config = hbeta_config or HbetaComplexConfig()
    mgii_config = mgii_config or MgIIComplexConfig()
    halpha_config = halpha_config or HalphaComplexConfig()
    lya_nv_config = lya_nv_config or LyaNVComplexConfig()
    uncertainty_config = uncertainty_config or UncertaintyConfig()
    galactic_extinction_config = (
        galactic_extinction_config or GalacticExtinctionConfig()
    )
    configuration = _configuration(
        run_host_decomp=run_host_decomp,
        template_root=template_root,
        template_file=template_file,
        host_fit_range=host_fit_range,
        host_config=host_config,
        galactic_extinction_config=galactic_extinction_config,
        global_config=global_config,
        hbeta_config=hbeta_config,
        mgii_config=mgii_config,
        halpha_config=halpha_config,
        lya_nv_config=lya_nv_config,
        uncertainty_config=uncertainty_config,
        complexes=complexes,
    )
    store = RunStore.create(
        run_directory,
        configuration=configuration,
        configuration_summary=_configuration_overrides(configuration),
        run_id=run_id,
        resume=resume,
    )
    if isinstance(input_data, SpectrumInput):
        descriptor = input_data
        spectrum_data = read_spectrum(
            descriptor.source,
            row_index=descriptor.row_index,
            redshift=descriptor.redshift,
            object_id=descriptor.object_id,
            reader=descriptor.reader,
            flux_unit=flux_unit,
            flux_scale=flux_scale,
        )
    elif isinstance(input_data, SpectrumData):
        descriptor = SpectrumInput(
            source=str(input_data.metadata.get("input_file", "in_memory")),
            object_id=object_id or input_data.object_id or input_data.targetid,
            redshift=redshift or input_data.redshift,
            reader="memory",
        )
        spectrum_data = input_data
    elif isinstance(input_data, Spectrum):
        prepared_spectrum = prepare_spectrum(
            input_data,
            galactic_extinction_config=galactic_extinction_config,
        )
        descriptor = SpectrumInput(
            source=prepared_spectrum.metadata.source or "in_memory",
            object_id=object_id,
            redshift=prepared_spectrum.z,
            reader="memory",
        )
        spectrum_data = SpectrumData(
            wave_obs=prepared_spectrum.wave_obs,
            flux=prepared_spectrum.flux,
            error=prepared_spectrum.err,
            mask=prepared_spectrum.mask,
            redshift=prepared_spectrum.z,
            object_id=object_id,
            ra=prepared_spectrum.metadata.ra,
            dec=prepared_spectrum.metadata.dec,
            metadata={
                "input_file": descriptor.source,
                "flux_unit": prepared_spectrum.flux_unit,
                "flux_scale": prepared_spectrum.flux_scale,
                "spectrum_metadata": (
                    prepared_spectrum.metadata.to_dict()
                ),
                "galactic_extinction": dict(
                    prepared_spectrum.metadata.galactic_extinction
                ),
            },
        )
    else:
        descriptor = SpectrumInput(
            source=str(Path(input_data).expanduser()),
            row_index=row_index,
            object_id=object_id,
            redshift=redshift,
            reader=reader,
        )
        spectrum_data = read_spectrum(
            descriptor.source,
            row_index=row_index,
            redshift=redshift,
            object_id=object_id,
            reader=reader,
            flux_unit=flux_unit,
            flux_scale=flux_scale,
        )
    result, actual_object_id = _fit_spectrum_data(
        spectrum_data,
        descriptor=descriptor,
        **_fit_options(
            run_host_decomp=run_host_decomp,
            template_root=template_root,
            template_file=template_file,
            host_fit_range=host_fit_range,
            host_config=host_config,
            galactic_extinction_config=galactic_extinction_config,
            global_config=global_config,
            hbeta_config=hbeta_config,
            mgii_config=mgii_config,
            halpha_config=halpha_config,
            lya_nv_config=lya_nv_config,
            uncertainty_config=uncertainty_config,
            complexes=complexes,
        ),
    )
    store.write_payload(
        workflow_payload(
            result,
            run_id=store.run_id,
            object_key=descriptor.object_key,
            object_id=actual_object_id,
            input_record={
                "source": descriptor.source,
                "row_index": descriptor.row_index,
                "reader": descriptor.reader,
                "metadata": dict(descriptor.metadata),
            },
        )
    )
    finalize_run(store)
    result.output_files = {
        "run_directory": str(store.path),
    }
    result.output_files["manifest"] = str(store.path / "manifest.json")
    if write_legacy_products:
        config = qa_plot_config or GlobalQAPlotConfig()
        files = write_global_line_products(
            result,
            str(store.path / "legacy" / actual_object_id),
            config,
        )
        result.output_files.update(files)
    elif write_qa:
        from ..io.qa import render_qa

        rendered = render_qa(
            store,
            object_ids=[actual_object_id],
            plot_config=qa_plot_config or GlobalQAPlotConfig(),
        )
        result.output_files.update(rendered.get(actual_object_id, {}))
        result.output_files = {
            key: value
            for key, value in result.output_files.items()
            if key in ("run_directory", "manifest", "main_qa")
        }
    return result


def _iter_inputs(
    inputs,
    *,
    row_indices,
    filter_expression,
    parquet_batch_size,
) -> Iterator[tuple[SpectrumInput, Optional[SpectrumData]]]:
    items = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    parquet_spectra = []
    manifest_descriptors = []
    remaining = []
    for item in items:
        if isinstance(item, SpectrumInput):
            manifest_descriptors.append(item)
            continue
        if not isinstance(item, (str, Path)):
            remaining.append(item)
            continue
        path = Path(item).expanduser()
        suffix = path.suffix.lower()
        if suffix == ".csv":
            manifest_descriptors.extend(read_input_manifest(str(path)))
        elif suffix == ".parquet":
            import pyarrow.parquet as pq

            columns = {name.lower() for name in pq.read_schema(path).names}
            has_wave = any(
                alias in columns
                for alias in ("wavelength", "wave", "lambda", "lam", "obs_wave")
            )
            has_flux = any(
                alias in columns for alias in ("flux", "flam", "flux_lambda")
            )
            if has_wave and has_flux:
                parquet_spectra.append(str(path))
            else:
                manifest_descriptors.extend(read_input_manifest(str(path)))
        else:
            remaining.append(item)
    if parquet_spectra:
        yield from scan_parquet_spectra(
            parquet_spectra,
            row_indices=row_indices,
            filter_expression=filter_expression,
            batch_size=parquet_batch_size,
        )
    for descriptor in manifest_descriptors:
        yield descriptor, None
    discoverable = [
        str(item)
        for item in remaining
        if isinstance(item, (str, Path))
    ]
    if discoverable:
        for descriptor in discover_fits_inputs(discoverable):
            yield descriptor, None


def fit_batch(
    inputs,
    run_directory: str,
    *,
    row_indices=None,
    filter_expression=None,
    parquet_batch_size: int = 128,
    task_size: int = 8,
    n_workers: Union[int, str] = "auto",
    num_shards: int = 1,
    shard_index: int = 0,
    run_host_decomp: bool = False,
    template_root: str = "~/tools/ppxf_data",
    template_file: str = "spectra_emiles_9.0.npz",
    host_fit_range=(3600.0, 7000.0),
    host_config=None,
    galactic_extinction_config: Optional[
        GalacticExtinctionConfig
    ] = None,
    global_config: Optional[GlobalContinuumConfig] = None,
    hbeta_config: Optional[HbetaComplexConfig] = None,
    mgii_config: Optional[MgIIComplexConfig] = None,
    halpha_config: Optional[HalphaComplexConfig] = None,
    lya_nv_config: Optional[LyaNVComplexConfig] = None,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    complexes: Optional[Sequence[Union[str, ComplexRecipe]]] = None,
    run_id: Optional[str] = None,
    resume: bool = True,
    retry_failures: bool = True,
    finalize: bool = True,
    compact_models: bool = False,
    write_legacy_products: bool = False,
) -> BatchResult:
    """Fit a Parquet or FITS sample with resumable process parallelism."""

    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards.")
    hbeta_config = hbeta_config or HbetaComplexConfig()
    mgii_config = mgii_config or MgIIComplexConfig()
    halpha_config = halpha_config or HalphaComplexConfig()
    lya_nv_config = lya_nv_config or LyaNVComplexConfig()
    uncertainty_config = uncertainty_config or UncertaintyConfig()
    galactic_extinction_config = (
        galactic_extinction_config or GalacticExtinctionConfig()
    )
    preflight_galactic_extinction(galactic_extinction_config)
    configuration = _configuration(
        run_host_decomp=run_host_decomp,
        template_root=template_root,
        template_file=template_file,
        host_fit_range=host_fit_range,
        host_config=host_config,
        galactic_extinction_config=galactic_extinction_config,
        global_config=global_config,
        hbeta_config=hbeta_config,
        mgii_config=mgii_config,
        halpha_config=halpha_config,
        lya_nv_config=lya_nv_config,
        uncertainty_config=uncertainty_config,
        complexes=complexes,
    )
    configuration["num_shards"] = int(num_shards)
    store = RunStore.create(
        run_directory,
        configuration=configuration,
        configuration_summary=_configuration_overrides(configuration),
        run_id=run_id,
        resume=resume,
    )
    completed = store.completed_keys() if resume else set()
    failed = store.failed_keys() if resume and not retry_failures else set()
    options = _fit_options(
        run_host_decomp=run_host_decomp,
        template_root=template_root,
        template_file=template_file,
        host_fit_range=host_fit_range,
        host_config=host_config,
        galactic_extinction_config=galactic_extinction_config,
        global_config=global_config,
        hbeta_config=hbeta_config,
        mgii_config=mgii_config,
        halpha_config=halpha_config,
        lya_nv_config=lya_nv_config,
        uncertainty_config=uncertainty_config,
        complexes=complexes,
    )
    iterator = _iter_inputs(
        inputs,
        row_indices=row_indices,
        filter_expression=filter_expression,
        parquet_batch_size=parquet_batch_size,
    )

    def selected():
        for descriptor, spectrum_data in iterator:
            digest = int(
                hashlib.sha256(descriptor.object_key.encode("utf-8")).hexdigest(),
                16,
            )
            if digest % num_shards != shard_index:
                continue
            if descriptor.object_key in completed or descriptor.object_key in failed:
                yield None
                continue
            yield _Task(
                descriptor=descriptor,
                spectrum_data=spectrum_data,
                run_directory=str(store.path),
                fit_options=options,
                legacy_output=write_legacy_products,
            )

    def grouped_tasks():
        parquet_group = []
        for task in selected():
            if task is None:
                yield None
                continue
            if task.spectrum_data is None:
                if parquet_group:
                    yield parquet_group
                    parquet_group = []
                yield [task]
                continue
            parquet_group.append(task)
            if len(parquet_group) >= max(int(task_size), 1):
                yield parquet_group
                parquet_group = []
        if parquet_group:
            yield parquet_group

    task_iterator = iter(grouped_tasks())
    worker_count = _auto_workers(None) if n_workers == "auto" else int(n_workers)
    worker_count = max(worker_count, 1)
    if worker_count > 1 and not _process_pool_available():
        worker_count = 1
    submitted = completed_count = failed_count = skipped_count = 0

    def handle(output):
        nonlocal completed_count, failed_count
        store.promote(output["staging"])
        if output["success"]:
            store.clear_failure(output["object_key"])
            completed_count += 1
        else:
            failed_count += 1

    if worker_count == 1:
        for task_group in task_iterator:
            if task_group is None:
                skipped_count += 1
                continue
            submitted += len(task_group)
            for output in _run_task_group(task_group):
                handle(output)
    else:
        _worker_initializer()
        import multiprocessing as mp

        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_initializer,
        ) as executor:
            pending = {}
            exhausted = False
            while pending or not exhausted:
                while len(pending) < worker_count * 2:
                    try:
                        task_group = next(task_iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    if task_group is None:
                        skipped_count += 1
                        continue
                    submitted += len(task_group)
                    future = executor.submit(_run_task_group, task_group)
                    pending[future] = task_group
                if pending:
                    done, _ = wait(
                        set(pending), return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        task_group = pending.pop(future)
                        try:
                            outputs = future.result()
                        except Exception as exception:
                            outputs = []
                            for task in task_group:
                                store.write_payload(
                                    _failure_payload(
                                        store, task.descriptor, exception
                                    )
                                )
                                failed_count += 1
                        for output in outputs:
                            handle(output)
    datasets = (
        finalize_run(store)
        if finalize and num_shards == 1 else {}
    )
    return BatchResult(
        run_directory=str(store.path),
        run_id=store.run_id,
        n_submitted=submitted,
        n_completed=completed_count,
        n_failed=failed_count,
        n_skipped=skipped_count,
        n_workers=worker_count,
        datasets=datasets,
    )
