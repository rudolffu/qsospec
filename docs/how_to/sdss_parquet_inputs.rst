SDSS and DESI Parquet inputs
============================

Download SDSS spectra with ``sparcli --backend sdss``. QSOSpec reads the
resulting Parquet through the same ``scan_parquet_spectra``, ``fit_batch``,
``plan_batch_resume``, run-store, and HostSED APIs used for DESI.

Provide an explicit fitting redshift (``redshift`` or ``z_optical_fit``) and a stable
``spectrum_key``. Choose the catalogue redshift, such as ``Z_SYS``, when
preparing the input. Common
optical metadata includes ``optical_survey``, ``optical_object_id``, catalogid,
release, run2d, coadd, observatory, MJD, source URL/checksum, and optical-redshift
provenance. Exact integer IDs survive scalar scanning, selective-row resume,
and saved object metadata. Existing DESI/SPARCL fields remain supported.

The input adapter supplies observed-vacuum wavelength and per-pixel Gaussian
``sigma_lambda`` or ``lsf_sigma_angstrom``. For DR20/v6_2_1, sparcli derives this from
WRESL FWHM; WDISP is retained independently by the acquisition layer. Resolution
marked non-object-specific/invalid is treated as approximate when assessing
host reliability. Optical spectra
retain their declared flux scale (SDSS: ``1e-17``). The fitting workflow
applies extinction correction according to its configuration.

Pass explicit spectral Parquet files to the reader, excluding scalar download
manifests.
