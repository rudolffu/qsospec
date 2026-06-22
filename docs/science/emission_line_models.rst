Emission-line models
====================

Recipes
-------

Each emission complex is an immutable recipe containing its fit window,
components, line IDs, roles, profile multiplicities, kinematic bounds,
coverage rules, and QA labels. Dedicated adapters implement Hβ/[O III],
Mg II, Hα, and Lyα/N V; generic variable projection handles the remaining
recipes.

Profiles and roles
------------------

- Broad components represent the summed broad-line profile used for flux,
  centroid, dispersion, numerical FWHM, and EW.
- Narrow components use narrower FWHM and velocity bounds.
- Wing components are optional outflow-like profiles selected only when the
  relevant recipe supports them.
- Fixed ratios and shared kinematics are encoded in recipe metadata.

For [O III], the default wing candidate must improve BIC by at least 20, have
flux S/N of at least 5, be at least twice as broad as the narrow core, and
have a centroid separated from the core by at least 150 km/s. The candidate
diagnostics and rejection reasons are retained in result metadata.

Current UV defaults
-------------------

- Lyα: two broad Gaussians; N V: one broad effective 1240.14 Å blend.
- C IV: three broad Gaussians at the unresolved 1549.06 Å blend, with
  velocities from -5000 to +3000 km/s.
- C III]: two broad Gaussians at 1908.73 Å with velocities within
  :math:`\pm2000` km/s.
- No narrow UV components are included by default.

Current optical/NIR defaults
----------------------------

Dedicated recipes fit Mg II, Hβ/[O III], and Hα/[N II]/[S II]. The
optical-blue and Paschen/NIR recipes are component-adaptive after the full
window passes its minimum coverage requirement.

See :doc:`../reference/recipes` for exact windows and components.
