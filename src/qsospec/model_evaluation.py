"""Versioned, optimizer-free evaluation of archived fitted components.

No pickle, executable imports, or untrusted callables are accepted by the codec.
"""

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
import hashlib
import json
import numpy as np

VERSION = 1


def array_identity(array):
    a = np.ascontiguousarray(array)
    if a.dtype.kind not in "biufSU":
        raise ValueError(f"Unsupported template asset dtype: {a.dtype}")
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def encode(value, assets):
    if isinstance(value, np.ndarray):
        token = array_identity(value)
        assets[token] = value
        return {"asset": token}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        name = type(value).__name__
        # Plain configuration/warning records need only attribute access.
        return {"record": name, "fields": {f.name: encode(getattr(value, f.name), assets) for f in fields(value)}}
    if isinstance(value, dict):
        return {str(k): encode(v, assets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, assets) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported reconstruction value: {type(value).__name__}")


def decode(value, assets):
    if isinstance(value, list):
        return [decode(v, assets) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"asset"}:
        return assets(value["asset"])
    if set(value) == {"record", "fields"}:
        from .templates.iron import IronTemplate
        from .templates.balmer import BalmerSeriesTemplate
        from .workflows.host.templates import PPXFTemplateLibrary
        from .workflows.host.ppxf_host import PreprocessedSpectrum
        from .resolution import SpectralResolution

        classes = {
            c.__name__: c
            for c in (IronTemplate, BalmerSeriesTemplate, PPXFTemplateLibrary, PreprocessedSpectrum, SpectralResolution)
        }
        state = {k: decode(v, assets) for k, v in value["fields"].items()}
        cls = classes.get(value["record"])
        return cls(**state) if cls else SimpleNamespace(**state)
    return {k: decode(v, assets) for k, v in value.items()}


def continuum_state(context):
    names = (
        "names",
        "index",
        "fixed_parameters",
        "config",
        "uv_template",
        "opt_template",
        "full_template",
        "middle_template",
        "balmer_template",
        "polynomial_enabled",
        "polynomial_pivot",
        "polynomial_scale",
        "bridge_parent",
        "bridge_intervals",
        "bridge_norm",
    )
    return {key: getattr(context, key) for key in names if hasattr(context, key)}


def evaluate_continuum(state, parameters, wave):
    from .fitting.global_fit import _ContinuumContext

    context = object.__new__(_ContinuumContext)
    context.__dict__.update(state)
    context._regional_weights_memo = None
    return context.components(np.array([parameters[k] for k in context.names]), wave)


def evaluate(recipe, wave, assets, row, memo=None):
    if recipe.get("version") != VERSION:
        raise ValueError(f"Unsupported model evaluator version: {recipe.get('version')}")
    memo = {} if memo is None else memo
    kind = recipe["kind"]
    if kind == "continuum":
        key = ("continuum", json.dumps([recipe["state"], recipe["parameters"]], sort_keys=True))
        if key not in memo:
            memo[key] = evaluate_continuum(decode(recipe["state"], assets), recipe["parameters"], wave)
        return memo[key][recipe["component"]]
    if kind == "line":
        from .fitting.complexes import _profile
        from .fitting.line_lsf import GaussianLineLSF

        parameters = recipe["parameters"]
        output = np.zeros_like(wave)
        for definition in decode(recipe["definitions"], assets):
            args = (
                wave,
                definition["reference_wave"],
                parameters[definition["velocity_parameter"]],
                parameters[definition["width_parameter"]],
                definition["profile"],
            )
            if definition.get("line_lsf"):
                descriptor = definition["line_lsf"]
                key = ("lsf", json.dumps(descriptor, sort_keys=True))
                if key not in memo:
                    memo[key] = GaussianLineLSF(descriptor, native_wave=wave)
                basis = memo[key].profile(*args, integrated=True)[0]
            else:
                basis = _profile(*args)[0]
            output += parameters[definition["flux_parameter"]] / definition["flux_divisor"] * basis
        return output
    if kind == "local_continuum":
        return (
            np.full_like(wave, recipe["value"]) if recipe["order"] == 0 else recipe["value"] * (wave - recipe["pivot"])
        )
    if kind == "reference":
        return np.asarray(row[recipe["column"]], dtype=float)
    if kind == "host":
        key = ("host", json.dumps(recipe["state"], sort_keys=True))
        if key not in memo:
            memo[key] = evaluate_host(decode(recipe["state"], assets), wave)
        return memo[key][recipe["component"]]
    raise ValueError(f"Unsupported model evaluator: {kind}")


def matches(original, recovered):
    a, b = np.asarray(original), np.asarray(recovered)
    return (
        a.shape == b.shape
        and np.array_equal(np.isnan(a), np.isnan(b))
        and np.array_equal(np.isposinf(a), np.isposinf(b))
        and np.array_equal(np.isneginf(a), np.isneginf(b))
        and np.allclose(a[np.isfinite(a)], b[np.isfinite(a)], rtol=1e-10, atol=1e-12)
    )


def compact_row(row, assets, external_assets=None):
    """Validate every recipe before discarding an array; retain explicit fallbacks."""
    from .io.run_store import _from_key_values, _key_values

    metadata = _from_key_values(row["workflow_metadata"])
    states = row.pop("_evaluation_states", {})
    wave = np.asarray(row["wave_rest"], dtype=float)
    complexes = {c["recipe_id"]: _from_key_values(c["metadata"]) for c in row["complexes"]}
    recipes = {}
    encoding_errors = {}
    if "continuum" in states:
        try:
            recipes["continuum"] = encode(states["continuum"], assets)
        except (ValueError, TypeError) as exc:
            encoding_errors["continuum"] = str(exc)
    host = None
    if "host" in states:
        try:
            host = encode(states["host"], assets)
        except (ValueError, TypeError) as exc:
            encoding_errors["host"] = str(exc)
    outcomes = []
    memo = {}

    def asset_lookup(token):
        return assets[token] if token in assets else external_assets(token)

    for component in row["components"]:
        existing_definition = component.get("definition")
        if existing_definition and component["values"] is None:
            component["values"] = evaluate(json.loads(existing_definition), wave, asset_lookup, row, memo)
        component["definition"] = None
        section, name = component["section"], component["name"]
        recipe = json.loads(existing_definition) if existing_definition else None
        if recipe:
            pass
        elif section == "continuum" and "continuum" in recipes:
            recipe = dict(
                kind="continuum",
                state=recipes["continuum"]["state"],
                parameters=recipes["continuum"]["parameters"],
                component=name,
            )
        elif section == "complex":
            fit = complexes[component["recipe_id"]]
            definitions = fit.get("peak_model", {}).get("components", [])
            selected = [d for d in definitions if d["component_id"] == name]
            params = states.get(component["recipe_id"], {}).get("parameters", {})
            if selected and params:
                recipe = dict(kind="line", definitions=encode(selected, assets), parameters=params)
            elif name.startswith("local_continuum_") and params:
                order = 0 if name.endswith("constant") else 1
                pivot = fit.get("local_continuum_pivot")
                if order == 0 or pivot is not None:
                    recipe = dict(
                        kind="local_continuum",
                        order=order,
                        value=params["continuum.constant" if order == 0 else "continuum.slope"],
                        pivot=pivot,
                    )
        elif section == "host":
            if name == "host_subtracted_flux":
                recipe = dict(kind="reference", column="flux")
            elif host:
                recipe = dict(kind="host", state=host, component=name)
        reason = "unsupported_or_missing_evaluation_state"
        if section in encoding_errors:
            reason = "unsupported_evaluation_state: " + encoding_errors[section]
        if recipe:
            recipe["version"] = VERSION
            try:
                if matches(component["values"], evaluate(recipe, wave, asset_lookup, row, memo)):
                    component["definition"] = json.dumps(recipe, sort_keys=True, allow_nan=True)
                    component["values"] = None
                    reason = "parameters"
                else:
                    reason = "reconstruction_mismatch"
            except (ValueError, TypeError, KeyError, AttributeError, ImportError) as exc:
                reason = f"reconstruction_unavailable: {exc}"
        outcomes.append(
            dict(
                section=section,
                recipe_id=component["recipe_id"],
                component=name,
                storage="parameters" if component["definition"] else "array_fallback",
                reason=reason,
            )
        )
    root_definitions = metadata.get("model_storage_root_definitions", {})
    if row.get("total_flux") is not None and matches(row["total_flux"], row["flux"]):
        root_definitions["total_flux"] = dict(version=VERSION, kind="reference", column="flux")
    if host and row.get("host_model") is not None:
        candidate = dict(version=VERSION, kind="host", state=host, component="__host_on_grid__")
        try:
            if matches(row["host_model"], evaluate(candidate, wave, asset_lookup, row, memo)):
                root_definitions["host_model"] = candidate
        except (ValueError, TypeError, AttributeError, KeyError, ImportError):
            pass
    for column, definition in root_definitions.items():
        if row.get(column) is not None and matches(row[column], evaluate(definition, wave, asset_lookup, row, memo)):
            row[column] = None
    metadata["model_storage_root_definitions"] = root_definitions
    metadata["model_storage_components"] = outcomes
    row["workflow_metadata"] = _key_values(metadata)
    return row


def restore_row(row, assets):
    memo = {}
    decoded_assets = {}

    def asset_lookup(token):
        if token not in decoded_assets:
            decoded_assets[token] = assets(token)
        return decoded_assets[token]

    wave = np.asarray(row["wave_rest"], dtype=float)
    for component in row["components"]:
        definition = component.get("definition")
        if definition and component["values"] is None:
            component["values"] = evaluate(json.loads(definition), wave, asset_lookup, row, memo)
        elif component["values"] is None:
            raise ValueError(f"Missing array and reconstruction definition: {component['name']}")
    from .io.run_store import _from_key_values

    for column, definition in (
        _from_key_values(row["workflow_metadata"]).get("model_storage_root_definitions", {}).items()
    ):
        if row.get(column) is None:
            row[column] = evaluate(definition, wave, asset_lookup, row, memo)
    return row


def evaluate_host(state, wave):
    """Replay saved pPXF template transforms and weights, never a linear solve."""
    from ppxf.ppxf import losvd_rfft, rebin
    from .workflows.host.ppxf_host import _resample_stellar_templates, _interpolate_component

    prep = state["preprocessed"]
    stellar, scales, _ = _resample_stellar_templates(prep, state["templates"])
    bundle = None
    if state.get("agn_recipe"):
        from .workflows.host.agn_templates import build_host_agn_template_bundle

        bundle = build_host_agn_template_bundle(prep.wave_log, **state["agn_recipe"])
    from .workflows.host.ppxf_host import _build_agn_basis

    agn_matrix = (
        bundle.matrix
        if bundle is not None
        else (
            _build_agn_basis(prep.wave_log, state["powerlaw_slopes"])
            if state.get("powerlaw_slopes")
            else state["agn_matrix"]
        )
    )
    matrix = np.column_stack([stellar, agn_matrix])
    s = state["transform"]
    fft = np.fft.rfft(matrix, s["npad"], axis=0)
    losvd = losvd_rfft(
        np.asarray(s["pars"]),
        1,
        np.asarray(s["moments"]),
        fft.shape[0],
        s["ncomp"],
        s["vsyst"],
        s["velscale_ratio"],
        s["sigma_diff"],
    )
    transformed = np.empty((s["npix"], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        transformed[:, j] = rebin(
            np.fft.irfft(fft[:, j] * losvd[:, s["component"][j], 0], s["npad"])[: s["npix"] * s["velscale_ratio"]],
            s["velscale_ratio"],
        )
    multiplicative = s.get("mpoly")
    if s.get("mpoly_coefficients") is not None:
        multiplicative = np.polynomial.legendre.legval(
            np.linspace(-1, 1, s["npix"]), [1.0, *s["mpoly_coefficients"]]
        ).clip(0.1)
    if multiplicative is not None:
        transformed *= np.asarray(multiplicative)[:, None]
    n = stellar.shape[1]
    host = transformed[:, :n] @ np.asarray(state["weights"][:n])
    agn = transformed[:, n:] @ np.asarray(state["weights"][n:])
    models = {"stellar": host, "agn_total": agn}
    groups = state["groups"]
    if bundle is not None:
        category = {
            "agn_powerlaw": "powerlaw",
            "agn_feii_optical": "feii_optical",
            "agn_feii_uv": "feii_uv",
            "agn_feii_middle": "middle_iron",
            "agn_feii_full": "feii_full",
            "agn_balmer_continuum": "balmer_continuum",
            "agn_balmer_high_order": "balmer_high_order",
        }
        groups = [
            dict(
                column=column,
                items=[
                    dict(name=category[item.category], values=item.values)
                    for item in bundle.components
                    if item.linear_group == name
                ],
            )
            for name, column in bundle.group_column_indices.items()
        ]
    if not groups:
        models["powerlaw"] = agn
    for group in groups:
        column = group["column"]
        raw = sum((i["values"] for i in group["items"]), np.zeros_like(host))
        model = transformed[:, n + column] * state["weights"][n + column]
        for item in group["items"]:
            fraction = np.divide(item["values"], raw, out=np.zeros_like(raw), where=np.abs(raw) > np.finfo(float).eps)
            key = item["name"]
            models[key] = models.get(key, np.zeros_like(host)) + model * fraction
    coefficients = state["additive_coefficients"]
    additive = (
        np.polynomial.legendre.legval(np.linspace(-1, 1, len(host)), coefficients)
        if coefficients
        else np.zeros_like(host)
    )
    # The fitting code includes multiplicative effects in transformed physical components.
    physical = host + agn + additive
    best_coefficients = state.get("bestfit_additive_coefficients", coefficients)
    best_additive = (
        np.polynomial.legendre.legval(np.linspace(-1, 1, len(host)), best_coefficients)
        if best_coefficients
        else np.zeros_like(host)
    )
    bestfit = host + agn + best_additive
    models.update(
        polynomial_additive=additive,
        polynomial_multiplicative_effect=np.zeros_like(host),
        physical_component_total=physical,
        ppxf_bestfit=bestfit,
        closure_residual=bestfit - physical,
    )
    result = {}
    for name, value in models.items():
        on_native = _interpolate_component(prep, value)
        valid = np.isfinite(prep.wave_rest) & np.isfinite(on_native)
        order = np.argsort(prep.wave_rest[valid])
        result[name] = np.interp(wave, prep.wave_rest[valid][order], on_native[valid][order], left=np.nan, right=np.nan)
    source = state["templates"]
    source_flux = (source.source_flux / scales[np.newaxis, :]) @ np.asarray(state["weights"][:n])
    source_flux *= prep.normalization
    projected = np.interp(wave, source.source_wave, source_flux, left=np.nan, right=np.nan)
    finite = np.isfinite(result["stellar"])
    projected[finite] = result["stellar"][finite]
    result["__host_on_grid__"] = projected
    return result
