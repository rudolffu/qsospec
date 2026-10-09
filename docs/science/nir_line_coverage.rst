NIR line coverage
=================

The ``extended_quasar_v1`` preset makes the complete emission-line inventory
available through a documented, versioned recipe set:

.. code-block:: python

   import qsospec

   complexes = qsospec.recipes.extended_quasar()
   result = qsospec.fit_global_lines(spectrum, complexes=complexes)

   assert result.metadata["complex_preset_id"] == "extended_quasar_v1"

The preset keeps the standard auto-enabled UV/optical recipes, adds compact
recipes for He I 5877, O I 8449, and the [S III] NIR doublet, and replaces the
``paschen_nir`` umbrella with four compact local NIR recipes. The resolved
recipe configuration is recorded in
``result.metadata["complex_preset_configuration"]``.

Feature coverage
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 24 54

   * - Feature
     - Canonical line IDs
     - Recipe in ``extended_quasar_v1``
   * - Mg II
     - ``mgii_blend``
     - ``mgii``
   * - [O II]
     - ``oii_3727``, ``oii_3730``
     - ``oii_nev_neiii_hgamma`` (the summed doublet is reported)
   * - [Ne III]
     - ``neiii_3870``
     - ``oii_nev_neiii_hgamma``
   * - Hγ
     - ``hgamma``
     - ``oii_nev_neiii_hgamma`` (soft-coupled to the Balmer series)
   * - Hβ
     - ``hbeta``
     - ``hbeta_oiii``
   * - [O III] 4960, 5008
     - ``oiii_4960``, ``oiii_5008``
     - ``hbeta_oiii`` (fixed 5008/4960 ratio)
   * - He I 5877
     - ``hei_5877``
     - ``hei5877``
   * - Hα
     - ``halpha``
     - ``halpha_nii_sii``
   * - O I 8449
     - ``oi_8449``
     - ``oi8449``
   * - [S III] 9071, 9533
     - ``siii_9071``, ``siii_9533``
     - ``siii_nir`` (shared narrow kinematics, independent fluxes)
   * - Paδ
     - ``padelta``
     - ``padelta``
   * - He I 10833
     - ``hei_10833``
     - ``hei10833_pgamma`` (joint fit with Paγ)
   * - Paγ
     - ``pagamma``
     - ``hei10833_pgamma``
   * - O I 11290
     - ``oi_11290``
     - ``oi11290``
   * - Paβ
     - ``pabeta``
     - ``pabeta``

Laboratory wavelengths
----------------------

The recipes use laboratory vacuum wavelengths. Object-specific
``lambda_obs`` values are measured wavelengths. The additional transitions are He I 5877.25 Å, O I 8448.68 Å,
[S III] 9071.1 Å and 9533.2 Å, Paη (n=10→3) 9017.384 Å, and
Paε (n=8→3) 9548.588 Å. The repeated 1200 km/s FWHM entries in the
source table are not used as fitted-width constraints.

The [Ne III] line is labeled 3868.58 Å in the originating line table while the registry
uses 3869.86 Å, the SDSS MaNGA vacuum wavelength. The offset is a
difference between the reference tables for the same transition. The
recipes use the registry value.

NIR blends and nuisance lines
-----------------------------

The ``siii_nir`` recipe fits two local windows (roughly 8950–9180 Å and
9390–9680 Å). Each [S III] line is measurable when the other is outside
coverage. Paη and Paε are included as compact nuisance hydrogen components
with kinematics distinct from the [S III] group. Higher Paschen
broad/narrow components are available but disabled in the preset.

Paζ (9231.546 Å) lies between the two windows, so its core is excluded;
the recipe records a possible broad-wing contamination warning. The
[S III] 9533 / Paε decomposition is a genuine blend at low resolution:
the fit persists the full covariance and a blend-quality flag, and
individual deblended errors are withheld when the two components are
unresolved or unreliable. The measured doublet ratio is not imposed.

Local versus envelope coverage
------------------------------

Local-support recipes decide coverage per fitting window using the valid
fraction of available native pixels after masks, a minimum valid count,
and valid support around the line center and core. A missing distant window
cannot reject a line that is locally covered. The component states
``observed``, ``truncated``, ``masked_core``, and ``not_observed`` are saved
in ``fit.metadata["component_coverage_status"]``. See
:doc:`coverage_reliability` for the exact distinctions. Peak support and both half-maximum crossings are
reported, and measured-window fluxes are distinguished from
model-extrapolated total fluxes. Recipes such as ``hbeta_oiii`` and ``mgii``
use strict full-window coverage.

``complexes=None`` selects the ``paschen_nir`` umbrella recipe. Do not request
it together with the compact NIR recipes; conflicting requests raise ``overlapping_complex_recipes``
because the recipes contain overlapping lines.
