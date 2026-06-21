Inspect a run bundle
====================

Open the run and read its authoritative tables:

.. code-block:: python

   run = qsospec.open_run("runs/sample")

   objects = run.read_table("objects").to_pandas()
   measurements = run.read_table("measurements").to_pandas()
   warnings = run.read_table("warnings").to_pandas()
   failures = run.read_table("failures").to_pandas()

Load one archived model without refitting:

.. code-block:: python

   model = qsospec.load_model(run, "object-id-or-object-key")
   print(model.summary())

Object IDs may be duplicated. Use the collision-free ``object_key`` when an ID
is ambiguous.

Build a wide catalog by first inspecting available section/quantity pairs:

.. code-block:: python

   print(
       measurements[["section", "recipe_id", "quantity"]]
       .drop_duplicates()
       .sort_values(["section", "recipe_id", "quantity"])
   )

Then pass the desired mappings to :func:`qsospec.build_science_catalog`.

Next: :doc:`../reference/run_bundles`.
