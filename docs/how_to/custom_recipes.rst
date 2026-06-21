Create a custom recipe
======================

For a group of narrow lines, start with the registry helper:

.. code-block:: python

   custom = qsospec.recipes.generic_narrow_lines(
       ["nev_3427", "neiii_3870"],
       id="my_narrow_complex",
       label="My narrow complex",
       fit_window=(3350.0, 3950.0),
       fit_windows=((3350.0, 3950.0),),
       qa_labels=("nev_3427", "neiii_3870"),
   )

   result = qsospec.fit_global_lines(
       spectrum,
       complexes=[custom],
   )

For full control, construct immutable :class:`qsospec.ComponentRecipe` and
:class:`qsospec.ComplexRecipe` objects. Define component roles, multiplicity,
velocity/FWHM bounds, required lines, coverage mode, fit windows, and backend.

Common failures
---------------

- Unknown line ID: inspect :func:`qsospec.lines.list`.
- Complex skipped: check total window coverage, valid pixels, edge margins,
  and required line centers.
- Component disabled: component-adaptive recipes retain only covered lines.

Next: :doc:`../reference/recipes` and
:doc:`../reference/api/configuration`.
