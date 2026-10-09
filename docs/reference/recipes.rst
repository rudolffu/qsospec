Recipe reference
================

Runtime discovery
-----------------

.. code-block:: python

   for recipe in qsospec.recipes.list_complexes():
       print(qsospec.recipes.describe(recipe.id))

Built-in auto-enabled recipes
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - ID
     - Window (Å)
     - Default model and behavior
   * - ``lya_nv``
     - 1150–1290
     - Two broad Lyα plus one broad N V effective blend; dedicated Lyα
       coverage classifier and adapter.
   * - ``civ``
     - 1450–1700
     - Three broad C IV blend Gaussians; full coverage; generic backend.
   * - ``ciii``
     - 1700–1970
     - Two broad C III] Gaussians; full coverage; generic backend.
   * - ``mgii``
     - 2700–2900
     - Two broad plus one narrow Mg II blend; dedicated adapter.
   * - ``oii_nev_neiii_hgamma``
     - 3380–4425
     - [Ne V], [O II], [Ne III], narrow and broad Hγ; component-adaptive
       generic backend.
   * - ``hbeta_oiii``
     - 4640–5100
     - Three broad Hβ, narrow Hβ/[O III], adaptive [O III] candidates and He II; shared generic
       engine by default, with an explicit legacy adapter.
   * - ``halpha_nii_sii``
     - 6400–6800
     - Three broad Hα plus narrow Hα/[N II]/[S II]; dedicated adapter.
   * - ``paschen_nir``
     - 9900–13050
     - Paδ, He I 10833, Paγ, O I 11290, and Paβ broad profiles;
       component-adaptive generic backend.

``generic_narrow_lines`` is available for custom construction but is not
auto-enabled.

extended-quasar preset
----------------------

``qsospec.recipes.extended_quasar()`` returns the versioned ``extended_quasar_v1``
recipe set (the full 17-feature emission-line inventory). It adds these compact recipes and
replaces ``paschen_nir`` with the compact NIR set:

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - ID
     - Window (Å)
     - Default model and behavior
   * - ``hei5877``
     - 5700–6050
     - Narrow and independently normalized broad He I 5877 components.
   * - ``oi8449``
     - 8200–8700
     - Broad O I 8449 plus a narrow component that is disabled by default;
       enable it with :meth:`ComplexRecipe.with_component`.
   * - ``siii_nir``
     - 8950–9180, 9390–9680
     - [S III] 9071/9533 with shared narrow kinematics and independent
       fluxes, plus Paη and Paε nuisance components; higher Paschen
       components are available but disabled.
   * - ``padelta``, ``oi11290``, ``pabeta``
     - local windows
     - One broad and one narrow Gaussian per line.
   * - ``hei10833_pgamma``
     - 10720–11040
     - Joint He I 10833 + Paγ broad (shared kinematics) and narrow profiles;
       amplitudes remain independent.

.. code-block:: python

   result = qsospec.fit_global_lines(
       spectrum, complexes=qsospec.recipes.extended_quasar()
   )

``complexes=None`` selects the ``paschen_nir`` umbrella for NIR coverage.
Requesting it together with the compact NIR recipes raises ``overlapping_complex_recipes``.

Coverage policy
---------------

Ordinary recipes apply their configured total-window overlap, valid-pixel
count, and required-center edge margin. The recipe defaults are 80%,
30 pixels, and 1000 km/s; built-ins can override them (Hα uses 60% overlap).
Component-adaptive recipes without local support choose covered components
only after their total window passes. See
:doc:`../science/coverage_reliability` for the coverage classes.

Recipes with ``local_support=True`` (the compact set) instead evaluate
each window against its valid fraction of available native pixels, minimum
pixel count, and center/core margins. A missing distant window cannot reject a locally
covered line; masked line cores are detected, and per-component status
(``observed``, ``truncated``, ``masked_core``, ``not_observed``) is
recorded in result metadata. See :doc:`../science/nir_line_coverage`.

Lyα fits full and red-side-only coverage. Red-side-only measurements are
limited and never reliable; edge-truncated and not-covered cases are skipped.

Custom recipes
--------------

Use :func:`qsospec.recipes.generic_narrow_lines` or construct
:class:`qsospec.ComponentRecipe` and :class:`qsospec.ComplexRecipe`. See
:doc:`../how_to/custom_recipes`.

Line registry
-------------

The line registry stores canonical vacuum wavelengths and aliases:

.. code-block:: python

   qsospec.lines.list()
   qsospec.lines.resolve("oiii 5007")
   qsospec.lines.get("oiii_5008")
