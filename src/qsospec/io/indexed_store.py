"""Transactional object locators and immutable bounded Parquet shards."""

from contextlib import contextmanager
from pathlib import Path
import json
import os
import sqlite3
from uuid import uuid4
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ..model_evaluation import array_identity


def write_parquet(table, path):
    temporary = Path(path).with_suffix(".schema.parquet")
    pq.write_table(table.slice(0, 0), temporary)
    schema = pq.ParquetFile(temporary).schema
    float_columns = [
        schema.column(i).path for i in range(len(schema)) if schema.column(i).physical_type in {"FLOAT", "DOUBLE"}
    ]
    temporary.unlink()
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
        use_byte_stream_split=float_columns or False,
    )
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


@contextmanager
def connection(store, write=False):
    path = store.path / "object-index.sqlite"
    db = sqlite3.connect(str(path) if write else f"file:{path}?mode=ro", uri=not write, timeout=60)
    try:
        if write:
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS locations (table_name TEXT, object_key TEXT, path TEXT, row_group INTEGER, PRIMARY KEY(table_name, object_key))"
            )
            db.execute("CREATE INDEX IF NOT EXISTS location_paths ON locations(path)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS pending_shards (path TEXT PRIMARY KEY, raw_bytes INTEGER NOT NULL DEFAULT 0)"
            )
            if "raw_bytes" not in {row[1] for row in db.execute("PRAGMA table_info(pending_shards)")}:
                db.execute("ALTER TABLE pending_shards ADD COLUMN raw_bytes INTEGER NOT NULL DEFAULT 0")
            db.execute("BEGIN IMMEDIATE")
        yield db
        if write:
            db.commit()
    except BaseException:
        if write:
            db.rollback()
        raise
    finally:
        db.close()


def locations(store, name, key=None):
    if not (store.path / "object-index.sqlite").exists():
        return []
    with connection(store) as db:
        if key is None:
            return db.execute(
                "SELECT object_key,path,row_group FROM locations WHERE table_name=? ORDER BY object_key", (name,)
            ).fetchall()
        return db.execute(
            "SELECT object_key,path,row_group FROM locations WHERE table_name=? AND object_key=?", (name, str(key))
        ).fetchall()


def publish(store, tables, replace_keys, prefix="pending"):
    """Commit locators for all tables together after all immutable files exist."""
    written = {}
    with connection(store, write=True) as db:
        for name, keys in replace_keys.items():
            db.executemany("DELETE FROM locations WHERE table_name=? AND object_key=?", [(name, k) for k in keys])
        for name, table in tables.items():
            keys = sorted(set(table["object_key"].to_pylist()))
            chunks, size, count = [], 0, 0

            def flush():
                nonlocal chunks, size, count
                if not chunks:
                    return
                path = store._table_path(name) / f"{prefix}-{uuid4().hex}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                with pq.ParquetWriter(
                    path,
                    table.schema,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=False,
                    use_byte_stream_split=_float_paths(table),
                ) as writer:
                    for _, chunk in chunks:
                        writer.write_table(chunk, row_group_size=max(1, chunk.num_rows))
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                sync_directory(path.parent)
                relative = str(path.relative_to(store.path))
                for group, (key, _) in enumerate(chunks):
                    db.execute("INSERT OR REPLACE INTO locations VALUES (?,?,?,?)", (name, key, relative, group))
                if prefix == "pending":
                    db.execute(
                        "INSERT OR IGNORE INTO pending_shards VALUES (?,?)",
                        (relative, sum(chunk.nbytes for _, chunk in chunks)),
                    )
                written[name] = str(path)
                chunks, size, count = [], 0, 0

            for key in keys:
                import pyarrow.compute as pc

                chunk = table.filter(pc.equal(table["object_key"], key))
                if chunks and (
                    count >= store.manifest.get("shard_objects", 128)
                    or size + chunk.nbytes > store.manifest.get("shard_bytes", 64 * 1024**2)
                ):
                    flush()
                chunks.append((key, chunk))
                size += chunk.nbytes
                count += 1
            flush()
    return written


def _float_paths(table):
    # Build the Parquet schema without persisting a helper file.
    sink = pa.BufferOutputStream()
    pq.write_table(table.slice(0, 0), sink)
    schema = pq.ParquetFile(pa.BufferReader(sink.getvalue())).schema
    return [
        schema.column(i).path for i in range(len(schema)) if schema.column(i).physical_type in {"FLOAT", "DOUBLE"}
    ] or False


def read_object(store, name, key, columns=None):
    if not (store.path / "object-index.sqlite").exists():
        return None
    with connection(store) as db:
        db.execute("BEGIN")
        found = db.execute(
            "SELECT path,row_group FROM locations WHERE table_name=? AND object_key=?", (name, str(key))
        ).fetchone()
        if not found:
            return None
        relative, group = found
        path = store.path / relative
        if path.resolve().parent != store._table_path(name).resolve():
            raise ValueError("Invalid object-index path")
        table = pq.ParquetFile(path).read_row_group(group, columns=columns)
        if "object_key" in table.column_names and set(table["object_key"].to_pylist()) != {str(key)}:
            raise ValueError("Object index points to inconsistent shard")
        return table


def read_table(store, name, columns=None):
    if not (store.path / "object-index.sqlite").exists():
        return None
    import pyarrow.compute as pc
    from .run_store import SCHEMAS

    with connection(store) as db:
        db.execute("BEGIN")
        found = db.execute(
            "SELECT object_key,path,row_group FROM locations WHERE table_name=? ORDER BY object_key", (name,)
        ).fetchall()
        if not found:
            return None
        grouped = {}
        for key, relative, group in found:
            grouped.setdefault(relative, []).append((key, group))
        tables = []
        requested = None if columns is None else list(dict.fromkeys([*columns, "object_key"]))
        for relative, entries in grouped.items():
            path = store.path / relative
            if path.resolve().parent != store._table_path(name).resolve():
                raise ValueError("Invalid object-index path")
            file = pq.ParquetFile(path)
            table = file.read_row_groups(sorted(set(group for _, group in entries)), columns=requested)
            table = table.filter(pc.is_in(table["object_key"], value_set=pa.array([key for key, _ in entries])))
            if columns is None and not table.schema.equals(SCHEMAS[name], check_metadata=False):
                table = pa.Table.from_pylist(table.to_pylist(), schema=SCHEMAS[name])
            tables.append(table.select(columns) if columns else table)
        return pa.concat_tables(tables, promote_options="default")


def collect_obsolete(store, candidates):
    # An exclusive SQLite transaction waits until snapshot readers finish using files.
    with sqlite3.connect(store.path / "object-index.sqlite", timeout=60) as db:
        db.execute("BEGIN EXCLUSIVE")
        active = {row[0] for row in db.execute("SELECT DISTINCT path FROM locations")}
        for relative in candidates:
            if relative not in active:
                (store.path / relative).unlink(missing_ok=True)
                db.execute("DELETE FROM pending_shards WHERE path=?", (relative,))


def compact(store, names=None, cleanup=True):
    """Stream bounded object groups into one atomically published generation."""
    from .run_store import TABLE_NAMES, SCHEMAS

    selected = names or TABLE_NAMES
    result = {}
    # Scalar key discovery does not read any model-array column.
    key_sets = {
        name: sorted(set(store.read_table(name, columns=["object_key"])["object_key"].to_pylist())) for name in selected
    }
    with connection(store, write=True) as db:
        for name in selected:
            schema = SCHEMAS[name]
            chunks, size = [], 0

            def flush():
                nonlocal chunks, size
                if not chunks:
                    return
                path = store._table_path(name) / f"shard-{uuid4().hex}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                with pq.ParquetWriter(
                    path,
                    schema,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=False,
                    use_byte_stream_split=_float_paths(chunks[0][1]),
                ) as writer:
                    for _, table in chunks:
                        writer.write_table(table, row_group_size=max(1, table.num_rows))
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                sync_directory(path.parent)
                relative = str(path.relative_to(store.path))
                for group, (key, _) in enumerate(chunks):
                    db.execute("INSERT OR REPLACE INTO locations VALUES (?,?,?,?)", (name, key, relative, group))
                result[name] = str(path)
                chunks, size = [], 0

            for key in key_sets[name]:
                if not store.manifest.get("indexed_storage") and name == "inputs":
                    import pyarrow.dataset as ds

                    table = store.read_table(name, filter_expression=ds.field("object_key") == key)
                else:
                    table = store.read_object_table(name, key)
                if not table.num_rows:
                    # Older runs can have multi-object input shards.
                    import pyarrow.dataset as ds

                    table = store.read_table(name, filter_expression=ds.field("object_key") == key)
                if not table.schema.equals(schema, check_metadata=False):
                    table = pa.Table.from_pylist(table.to_pylist(), schema=schema)
                if chunks and (
                    len(chunks) >= store.manifest.get("shard_objects", 128)
                    or size + table.nbytes > store.manifest.get("shard_bytes", 64 * 1024**2)
                ):
                    flush()
                chunks.append((key, table))
                size += table.nbytes
            flush()
    store.manifest["indexed_storage"] = True
    store._write_manifest(reconcile=False)
    if cleanup:
        candidates = [
            str(path.relative_to(store.path))
            for name in TABLE_NAMES
            for path in store._table_path(name).glob("*.parquet")
        ]
        collect_obsolete(store, candidates)
    return result


def write_asset(store, token, array):
    if array_identity(array) != token:
        raise ValueError("Asset content hash mismatch")
    directory = store.path / "assets"
    directory.mkdir(exist_ok=True)
    path = directory / f"{token}.parquet"
    if path.exists():
        return
    a = np.asarray(array)
    table = pa.table({"values": pa.array(a.ravel())}).replace_schema_metadata(
        {b"shape": json.dumps(a.shape).encode(), b"dtype": a.dtype.str.encode(), b"sha256": token.encode()}
    )
    temporary = directory / f"{token}-{uuid4().hex}.tmp"
    write_parquet(table, temporary)
    os.replace(temporary, path)
    sync_directory(directory)


def read_asset(store, token):
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Invalid template asset identity")
    path = store.path / "assets" / f"{token}.parquet"
    if not path.is_file():
        raise ValueError(f"Missing reconstruction asset: {token}")
    table = pq.read_table(path)
    metadata = table.schema.metadata
    a = np.asarray(table["values"].to_numpy(), dtype=metadata[b"dtype"].decode()).reshape(
        json.loads(metadata[b"shape"])
    )
    if array_identity(a) != token:
        raise ValueError(f"Corrupt reconstruction asset: {token}")
    a.setflags(write=False)
    return a


def compact_pending(store):
    """Bound the coordinator's uncompacted records without reading full runs."""
    from .run_store import TABLE_NAMES, SCHEMAS

    with connection(store) as db:
        pending = db.execute(
            "SELECT l.table_name,l.object_key,l.path,l.row_group FROM pending_shards p JOIN locations l ON l.path=p.path"
        ).fetchall()
    keys = {key for _, key, _, _ in pending}
    paths = {path for _, _, path, _ in pending}
    totals = {}
    with connection(store) as db:
        sizes = dict(db.execute("SELECT path,raw_bytes FROM pending_shards"))
    for name, _, path, _ in pending:
        totals.setdefault(name, set()).add(path)
    total = max(
        (
            sum(
                sizes.get(path, 0)
                or sum(
                    pq.ParquetFile(store.path / path).metadata.row_group(i).total_byte_size
                    for i in range(pq.ParquetFile(store.path / path).metadata.num_row_groups)
                )
                for path in selected
            )
            for selected in totals.values()
        ),
        default=0,
    )
    if len(keys) < store.manifest.get("shard_objects", 128) and total < store.manifest.get("shard_bytes", 64 * 1024**2):
        return
    tables, replacements = {}, {}
    for name in TABLE_NAMES:
        selected = [key for table, key, _, _ in pending if table == name]
        if selected:
            tables[name] = pa.concat_tables(
                [store.read_object_table(name, key) for key in selected], promote_options="default"
            )
            if not tables[name].schema.equals(SCHEMAS[name], check_metadata=False):
                tables[name] = pa.Table.from_pylist(tables[name].to_pylist(), schema=SCHEMAS[name])
            replacements[name] = selected
    publish(store, tables, replacements, prefix="shard")
    collect_obsolete(store, paths)


def snapshot_read(function):
    """Hold one SQLite snapshot through a multi-table public model reload."""
    from functools import wraps

    @wraps(function)
    def wrapped(run, *args, **kwargs):
        from .run_store import RunStore

        store = RunStore.open(str(run)) if not isinstance(run, RunStore) else run
        if not (store.path / "object-index.sqlite").exists():
            return function(store, *args, **kwargs)
        with connection(store) as db:
            db.execute("BEGIN")
            db.execute("SELECT table_name FROM locations LIMIT 1").fetchone()
            return function(store, *args, **kwargs)

    return wrapped


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def coordinator(function):
    """Serialize coordinators; worker staging and snapshot readers remain independent."""
    from contextvars import ContextVar
    from functools import wraps

    active = ContextVar(f"coordinator_{function.__name__}", default=frozenset())

    @wraps(function)
    def wrapped(store, *args, **kwargs):
        identity = str(store.path.resolve())
        if identity in active.get():
            return function(store, *args, **kwargs)
        with (store.path / ".storage.lock").open("a+") as handle:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            token = active.set(active.get() | {identity})
            try:
                return function(store, *args, **kwargs)
            finally:
                active.reset(token)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return wrapped
