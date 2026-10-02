"""Build an exact object/grid/LSF-specific XSL host-template product."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsospec.workflows.host.io import read_sparcli_spectrum
from qsospec.workflows.host.ppxf_host import prepare_spectrum_for_host_decomp
from qsospec.workflows.host.preconvolved_templates import (
    build_preconvolved_xsl_product,
    preconvolution_cache_key,
    preconvolution_contract,
)
from qsospec.workflows.host.templates import load_ppxf_npz_templates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", required=True)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--object-key", required=True)
    parser.add_argument("--redshift", type=float)
    parser.add_argument("--template-root", required=True)
    parser.add_argument(
        "--source-template-file", default="spectra_xsl_9.0.npz"
    )
    parser.add_argument("--fit-range", nargs=2, type=float, default=(3600.0, 7000.0))
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    fit_range = (float(args.fit_range[0]), float(args.fit_range[1]))
    spectrum = read_sparcli_spectrum(
        args.spectrum,
        row_index=args.row_index,
        redshift=args.redshift,
        object_id=args.object_key,
    )
    prepared = prepare_spectrum_for_host_decomp(
        spectrum,
        redshift=args.redshift,
        fit_range=fit_range,
    )
    prepared.metadata["spectral_resolution"] = spectrum.resolution
    prepared.metadata["object_id"] = args.object_key
    prepared.metadata["host_preconvolution_object_key"] = args.object_key
    source = load_ppxf_npz_templates(
        template_root=args.template_root,
        template_file=args.source_template_file,
        template_family="xsl",
        template_profile="xsl_native",
        write_report=False,
    )
    contract = preconvolution_contract(
        source,
        prepared,
        object_key=args.object_key,
        fit_range=fit_range,
    )
    key = preconvolution_cache_key(contract)
    output = Path(args.cache_root).expanduser() / key[:2] / f"xsl_preconvolved_{key}.npz"
    product = build_preconvolved_xsl_product(
        source,
        prepared,
        output_path=output,
        object_key=args.object_key,
        fit_range=fit_range,
        overwrite=args.overwrite,
    )
    print(product.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
