Local fitting
=============

Local fitting treats each wavelength window independently. It is useful for
targeted measurements, exploratory work, and spectra that do not support a
global continuum.

.. code-block:: python

   config = qsospec.LocalFitConfig(
       windows=[
           qsospec.recipes.local_mgii(),
           qsospec.recipes.local_hbeta(profile="lorentzian"),
       ]
   )
   result = qsospec.fit_local(spectrum, config)

Each entry in ``result.window_results`` has independent success, warnings,
parameters, model arrays, and metrics. Failure in one window does not prevent
other windows from completing.

One complex
-----------

.. code-block:: python

   window = qsospec.recipes.local_hbeta(
       profile="gaussian",
       iron_template=qsospec.IronTemplateConfig.park22(),
   )
   fit = qsospec.fit_line_complex(spectrum, window)

Trade-offs
----------

Local fits do not share continuum or kinematic information across windows.
Their local continuum and iron components can therefore differ from the
global decomposition. Prefer global fitting for homogeneous multi-line
science measurements.
