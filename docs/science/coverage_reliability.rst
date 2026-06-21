Coverage and reliability
========================

General complex coverage
------------------------

Every complex requires at least 80% geometric overlap with its declared
window, enough valid pixels, and required line centers inside the configured
edge margin. Below this threshold the complete complex is ``not_covered``.

Component-adaptive recipes apply their per-line selection only after the
total window passes the threshold. A partial optical-blue fit remains
archived, but its QA zoom is shown only when [Ne V] 3427, both [O II] lines,
[Ne III] 3870, and Hγ are covered.

Lyα/N V
-------

Lyα has four coverage states:

``full``
   Coverage reaches the required blue and red limits with enough geometric
   overlap and valid pixels.

``red_side_only``
   The Lyα center and complete 1216–1290 Å red side are usable.

``edge_truncated``
   Useful overlap exists, but the center or complete red side is unsafe.

``not_covered``
   The overlap or valid-pixel count is insufficient.

Full and red-side-only cases are fitted. Edge-truncated and not-covered cases
are skipped. After a preliminary fit, contiguous residual runs below
:math:`-3\sigma` and no wider than 2000 km/s are masked and the complex is
refitted once.

``lya_fit_reliable`` is true only for a successful full fit with broad Lyα
flux S/N at least three, at most 20% absorption-masked pixels, available
covariance, and no active Lyα kinematic bound. Red-side-only measurements are
always retained as limited and unreliable.

Host decomposition
------------------

Host decomposition is enabled only for finite :math:`z < 1.2`. Host fractions
outside observed rest-frame coverage are not reported as constrained
measurements.
