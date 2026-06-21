Plotting, recipes, models, and solvers
======================================

.. currentmodule:: qsospec

Plotting
--------

.. autofunction:: plot_line_result
.. autofunction:: plot_local_result
.. autofunction:: save_local_window_plots

Registries
----------

.. autofunction:: qsospec.recipes.list_complexes
.. autofunction:: qsospec.recipes.get
.. autofunction:: qsospec.recipes.resolve
.. autofunction:: qsospec.recipes.describe
.. autofunction:: qsospec.recipes.generic_narrow_lines
.. autofunction:: qsospec.lines.list
.. autofunction:: qsospec.lines.get
.. autofunction:: qsospec.lines.resolve

Models and solvers
------------------

.. autofunction:: qsospec.models.gaussian
.. autofunction:: qsospec.models.gaussian_partials
.. autofunction:: qsospec.models.lorentzian
.. autofunction:: qsospec.models.lorentzian_partials
.. autofunction:: qsospec.models.continuum
.. autofunction:: qsospec.models.continuum_partials
.. autofunction:: qsospec.solvers.solve_variable_projection
.. autofunction:: qsospec.solvers.run_least_squares
