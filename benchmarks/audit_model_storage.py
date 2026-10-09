"""Lossless encoding and sparse-array prototypes for archived model shards.

Prototype schemas are benchmark artifacts, not readable production run stores.
Every variant is checked against the original float64 values and metadata.
The optional float32 prototype explicitly permits model-array rounding only.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def equivalent(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


def sparse_table(table):
    rows = deepcopy(table.to_pylist())
    original_size = retained_size = 0
    for row in rows:
        for component in row["components"]:
            array = np.asarray(component["values"], dtype=float)
            nonzero = np.flatnonzero(array != 0.0)  # NaNs also remain stored.
            start = int(nonzero[0]) if len(nonzero) else 0
            stop = int(nonzero[-1] + 1) if len(nonzero) else 0
            component.update(values=array[start:stop].tolist(), start=start, full_length=len(array))
            original_size += len(array)
            retained_size += stop - start
    schema = table.schema
    index = schema.get_field_index("components")
    component_type = schema.field("components").type.value_type
    sparse_type = pa.list_(
        pa.struct(list(component_type) + [pa.field("start", pa.int32()), pa.field("full_length", pa.int32())])
    )
    schema = schema.set(index, pa.field("components", sparse_type))
    return pa.Table.from_pylist(rows, schema=schema), original_size, retained_size


def restore_sparse(rows):
    for row in rows:
        for component in row["components"]:
            start = component.pop("start")
            full = np.zeros(component.pop("full_length"))
            full[start : start + len(component["values"])] = component["values"]
            component["values"] = full.tolist()
    return rows


def float32_components(table, target, numeric_paths):
    schema = table.schema
    fields = [
        pa.field(field.name, pa.list_(pa.float32())) if field.name == "values" else field
        for field in schema.field("components").type.value_type
    ]
    schema = schema.set(schema.get_field_index("components"), pa.field("components", pa.list_(pa.struct(fields))))
    prototype = table.cast(schema)
    pq.write_table(
        prototype,
        target,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        use_byte_stream_split=numeric_paths,
    )
    original, loaded = table.to_pylist()[0], pq.read_table(target).to_pylist()[0]
    for name in ("wave_rest", "flux", "error", "total_flux", "host_model", "workflow_metadata"):
        assert equivalent(original[name], loaded[name]), name
    error = np.asarray(original["error"])
    valid = np.isfinite(error) & (error > 0)
    bound = np.zeros_like(error)
    maximum = 0.0
    for a, b in zip(original["components"], loaded["components"]):
        values, rounded = np.asarray(a["values"]), np.asarray(b["values"])
        np.testing.assert_array_equal(np.isfinite(values), np.isfinite(rounded))
        difference = np.where(np.isfinite(values), np.abs(values - rounded), 0.0)
        bound += difference
        maximum = max(maximum, float(np.max(difference[valid] / error[valid])))
    return dict(
        object_key=original["object_key"],
        bytes=target.stat().st_size,
        maximum_component_error_in_pixel_sigma=maximum,
        maximum_sum_error_in_pixel_sigma=float(np.max(bound[valid] / error[valid])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pack", action="store_true")
    parser.add_argument(
        "--include-lossy",
        action="store_true",
        help="Measure float32 component rounding; never modify the source bundle.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.run.resolve()),
        "lossless_variants_verified": True,
        "production_schema_changed": False,
        "variants": {},
        "compressed_bytes_by_column": {},
        "spectra": [],
    }
    columns = Counter()
    arrays = Counter()
    packed_tables = []
    rounded_results = []
    for path in sorted((args.run / "data/models").glob("*.parquet")):
        table = pq.read_table(path)
        original = table.to_pylist()
        if args.pack:
            packed_tables.append(table)
        file = pq.ParquetFile(path)
        float_paths = [
            file.schema.column(i).path
            for i in range(len(file.schema))
            if file.schema.column(i).physical_type in ("DOUBLE", "FLOAT")
        ]
        if args.include_lossy:
            folder = args.output / "float32_components"
            folder.mkdir(exist_ok=True)
            rounded_results.append(float32_components(table, folder / path.name, float_paths))
        for group in range(file.metadata.num_row_groups):
            for index in range(file.metadata.row_group(group).num_columns):
                column = file.metadata.row_group(group).column(index)
                columns[column.path_in_schema.split(".")[0]] += column.total_compressed_size
        row = original[0]
        component_values = sum(len(c["values"]) for c in row["components"])
        arrays["component_float64_values"] += component_values
        exact_duplicates = []
        seen = {}
        for name, values in [(k, row[k]) for k in ("wave_rest", "flux", "error", "total_flux", "host_model")] + [
            (f"component:{i}:{c['name']}", c["values"]) for i, c in enumerate(row["components"])
        ]:
            if values is None:
                continue
            array = np.asarray(values, dtype=np.float64)
            fingerprint = hashlib.sha256(array.tobytes()).hexdigest()
            if fingerprint in seen:
                exact_duplicates.append([name, seen[fingerprint]])
            else:
                seen[fingerprint] = name
        variants = {
            "zstd_default": (table, dict(compression="zstd")),
            "zstd9_dictionary": (table, dict(compression="zstd", compression_level=9)),
            "zstd3_byte_stream_split": (
                table,
                dict(compression="zstd", compression_level=3, use_dictionary=False, use_byte_stream_split=float_paths),
            ),
            "zstd9_byte_stream_split": (
                table,
                dict(compression="zstd", compression_level=9, use_dictionary=False, use_byte_stream_split=float_paths),
            ),
        }
        sparse, full_count, retained_count = sparse_table(table)
        variants["sparse_zstd9"] = (
            sparse,
            dict(compression="zstd", compression_level=9, use_dictionary=False, use_byte_stream_split=float_paths),
        )
        item = dict(
            object_key=row["object_key"],
            z=row["redshift"],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_file_bytes=path.stat().st_size,
            component_values=component_values,
            sparse_component_values=retained_count,
            exact_duplicate_arrays=exact_duplicates,
            workflow_metadata_json_bytes=sum(len(v["value"]) for v in row["workflow_metadata"]),
            variants={},
        )
        for name, (variant, options) in variants.items():
            folder = args.output / name
            folder.mkdir(exist_ok=True)
            target = folder / path.name
            start = perf_counter()
            pq.write_table(variant, target, **options)
            write = perf_counter() - start
            start = perf_counter()
            loaded = pq.read_table(target).to_pylist()
            read = perf_counter() - start
            if name.startswith("sparse"):
                loaded = restore_sparse(loaded)
            assert equivalent(original, loaded), (path, name, "round-trip mismatch")
            item["variants"][name] = dict(
                bytes=target.stat().st_size, write_seconds=write, read_and_materialize_seconds=read, roundtrip="exact"
            )
        report["spectra"].append(item)
        print(
            path.name,
            "original",
            path.stat().st_size,
            "bss9",
            item["variants"]["zstd9_byte_stream_split"]["bytes"],
            flush=True,
        )
    for name in variants:
        entries = [s["variants"][name] for s in report["spectra"]]
        report["variants"][name] = dict(
            bytes=sum(v["bytes"] for v in entries),
            write_seconds=sum(v["write_seconds"] for v in entries),
            read_seconds=sum(v["read_and_materialize_seconds"] for v in entries),
        )
    report["compressed_bytes_by_column"] = dict(columns)
    report["component_value_counts"] = dict(arrays)
    report["archived_model_bytes"] = sum(s["original_file_bytes"] for s in report["spectra"])
    if args.pack:
        combined = pa.concat_tables(packed_tables)
        output = args.output / "packed_zstd3_bss.parquet"
        start = perf_counter()
        pq.write_table(
            combined,
            output,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            use_byte_stream_split=float_paths,
            row_group_size=len(packed_tables),
        )
        elapsed = perf_counter() - start
        assert equivalent(combined.to_pylist(), pq.read_table(output).to_pylist())
        report["packed_objects"] = dict(
            bytes=output.stat().st_size, write_seconds=elapsed, objects=len(packed_tables), roundtrip="exact"
        )
    if rounded_results:
        report["float32_components_prototype"] = dict(
            lossy=True,
            production_enabled=False,
            input_flux_error_wavelength_unchanged=True,
            scalar_metadata_unchanged=True,
            bytes=sum(v["bytes"] for v in rounded_results),
            maximum_component_error_in_pixel_sigma=max(
                v["maximum_component_error_in_pixel_sigma"] for v in rounded_results
            ),
            maximum_sum_error_in_pixel_sigma=max(v["maximum_sum_error_in_pixel_sigma"] for v in rounded_results),
            spectra=rounded_results,
        )
    (args.output / "storage.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
