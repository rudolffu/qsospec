Fit independent line windows
============================

Local fitting estimates a separate continuum and line model in each selected
window. Use it for a targeted measurement or when your spectrum does not
support a shared global continuum.

Prepare ``spectrum`` with the :doc:`array workflow <../how_to/fit_arrays>`
before the following calls. Local fitters require rest-frame-normalized flux
and do not perform foreground correction themselves.

Fit several windows
-------------------

.. code-block:: python

   import qsospec

   config = qsospec.LocalFitConfig(
       windows=[
           qsospec.recipes.local_mgii(),
           qsospec.recipes.local_hbeta(profile="lorentzian"),
       ]
   )
   result = qsospec.fit_local(spectrum, config)
   for window_id, fit in result.window_results.items():
       print(window_id, fit.success)
       print(fit.to_table())

Each entry in ``window_results`` has its own success flag, warnings,
parameters, models and component measurement table. Failure in one window
does not prevent other windows from completing.

Fit one complex
---------------

.. code-block:: python

   window = qsospec.recipes.local_hbeta(
       profile="gaussian",
       iron_template="park22",
   )
   fit = qsospec.fit_line_complex(spectrum, window)
   print(fit.success)
   print(fit.to_table())

Interpret local measurements
----------------------------

Local ``FitResult`` objects expose ``param_values`` and ``to_table()``;
global emission-complex results expose ``metrics`` and ``metric_errors``.
The local table gives one row per fitted component. Its ``center`` and
``fwhm`` are rest-frame Angstrom, and ``line_flux_input`` is integrated flux
in the input scaling. These component widths are distinct from a summed
broad-profile width in km/s.

Local fits do not share continuum or kinematic information between windows.
Their continuum and iron solutions can differ from the global decomposition,
so compare the same quantity and continuum convention across analyses.
For a shared continuum across several lines, follow :doc:`global_fitting`.
See :doc:`../reference/api/results` for local result types and
:doc:`../reference/recipes` for available windows.
