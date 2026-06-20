Recipe Reference
================

qsospec uses an immutable recipe system to define emission-line complexes.
Each :class:`~qsospec.ComplexRecipe` specifies the lines, components,
windows, masks, continuum policy, and coverage rules for one complex.

Listing available recipes
-------------------------

.. code-block:: python

    >>> qsospec.recipes.list_complexes()
    [<ComplexRecipe 'hb'>, <ComplexRecipe 'ha'>, <ComplexRecipe 'mgii'>, ...]

.. code-block:: python

    >>> qsospec.recipes.describe("hb")
    {'id': 'hb', 'label': 'Hβ / [O III]', 'aliases': [...], ...}


Built-in recipes
----------------

Hβ / [O III] (``hb``)
^^^^^^^^^^^^^^^^^^^^^^

- **Wavelength region**: 4200–5600 Å rest-frame
- **Components**: three ordered broad Hβ Gaussians, narrow Hβ + [O III]
  λ4959 + [O III] λ5007 (shared narrow kinematics), optional He II λ4686
- **Continuum**: fixed global (uses the pre-subtracted continuum from the
  global fit)
- **Notes**: uses dedicated adapter model with BIC-based [O III] wing selection

Mg II (``mgii``)
^^^^^^^^^^^^^^^^

- **Wavelength region**: 2200–3100 Å rest-frame
- **Components**: two ordered broad + one narrow Mg II λλ2796,2803
  (fixed doublet ratio)
- **Continuum**: fixed global

Hα / [N II] / [S II] (``ha``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Wavelength region**: 6300–6800 Å rest-frame
- **Components**: three ordered broad Hα, narrow Hα + [N II] λ6548 + [N II]
  λ6585 (shared narrow kinematics, fixed [N II] ratio), independent [S II]
  λ6716,6731
- **Continuum**: fixed global

Lyα / N V (``lya_nv``)
^^^^^^^^^^^^^^^^^^^^^^^^

- **Wavelength region**: 1100–1290 Å rest-frame
- **Components**: configurable number of broad + narrow Lyα Gaussians, N V as
  an effective blend or equal-doublet
- **Continuum**: Lyα-safe global (avoids anchoring on the forest or line peak)
- **Notes**: uses a dedicated coverage classifier and absorption-masking refit;
  records ``lya_coverage_status`` and ``lya_fit_reliable`` in metadata

C IV (``civ``)
^^^^^^^^^^^^^^

- **Wavelength region**: ~1400–1700 Å rest-frame
- **Components**: broad + narrow C IV λ1549
- **Continuum**: fixed global
- **Status**: initial generic implementation; C IV blue-wing/outflow model
  selection is deferred

C III] (``ciii``)
^^^^^^^^^^^^^^^^^

- **Wavelength region**: ~1800–2000 Å rest-frame
- **Components**: C III] λ1909 blend
- **Continuum**: fixed global

[O II] / Hγ / Ne III (``oii``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Wavelength region**: ~3650–4000 Å rest-frame
- **Components**: [O II] λλ3726,3729, Hγ λ4341, [Ne III] λ3869
- **Continuum**: fixed global

Paschen / NIR (``k_nir``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Wavelength region**: ~8200–11000 Å rest-frame
- **Components**: Paschen series lines, He I λ10833, Paγ λ10941
- **Continuum**: fixed global
- **Notes**: component-adaptive coverage; requires rest-frame NIR coverage


Recipe selection
----------------

By default (``complexes=None``), qsospec selects all covered, auto-enabled
recipes.  Hβ, Mg II, and Hα are also auto-enabled but require their dedicated
configurations to be passed.

Recipes are selected by coverage:

- The recipe's ``fit_windows`` are checked against the rest-frame spectrum.
- Components whose required lines are not covered are disabled.
- Recipes with insufficient coverage are skipped with a ``"coverage_insufficient"`` status.

The recipe system is extensible: you can create custom recipes with
:func:`qsospec.recipes.generic_narrow_lines` or by constructing
:class:`~qsospec.ComplexRecipe` and :class:`~qsospec.ComponentRecipe` objects
directly.


Line registry
-------------

.. automodule:: qsospec.lines
   :members: LineDefinition, list, resolve, get
   :noindex:

The line registry provides canonical vacuum-wavelength definitions for ~30
emission lines, including:

- Hydrogen Balmer series: Hα λ6565, Hβ λ4863, Hγ λ4342, Hδ λ4103
- Helium: He I λ10833, He II λ4686
- Forbidden lines: [O III] λλ4959,5007, [N II] λλ6548,6585, [S II] λλ6716,6731
- UV resonance lines: Lyα λ1216, N V λλ1238,1242, C IV λ1549, C III] λ1909
- Mg II λλ2796,2803, Fe II (various multiples)

Resolve lines by canonical ID or alias:

.. code-block:: python

    >>> qsospec.lines.resolve("oiii 5007")
    'oiii_5007'
    >>> line = qsospec.lines.get("oiii_5007")
    >>> line.vacuum_wavelength
    5008.240
