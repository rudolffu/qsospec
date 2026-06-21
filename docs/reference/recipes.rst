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
     - Three broad Hβ, narrow Hβ/[O III], optional wings and He II; dedicated
       adapter.
   * - ``halpha_nii_sii``
     - 6400–6800
     - Three broad Hα plus narrow Hα/[N II]/[S II]; dedicated adapter.
   * - ``paschen_nir``
     - 9900–13050
     - Paδ, He I 10833, Paγ, O I 11290, and Paβ broad profiles;
       component-adaptive generic backend.

``generic_narrow_lines`` is available for custom construction but is not
auto-enabled.

Coverage policy
---------------

General recipes require at least 80% total-window overlap, enough valid
pixels, and safe required-line centers. Component-adaptive recipes choose
covered components only after the total window passes.

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
