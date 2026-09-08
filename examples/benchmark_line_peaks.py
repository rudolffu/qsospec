"""Benchmark peak recording and WS22 on extracted spectra without post-fit optimizers.

Use extract_archived_comparison.py to prepare the input JSONL. The archived
host is held fixed. This does not update the source run or adopted redshift.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
import qsospec
from qsospec.line_peaks import recover_line_peaks
from qsospec.metadata import resolve_spectrum_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    reports=[]
    for line in Path(args.input).read_text().splitlines():
        row=json.loads(line)
        spectrum=qsospec.Spectrum.from_arrays(row['wave_rest'],row['flux'],err=row['error'],
            z=row['z'],wave_frame='rest',mask=row['mask'],metadata=resolve_spectrum_metadata(metadata=row['spectrum_metadata']))
        start=perf_counter()
        result=qsospec.fit_global_lines(spectrum,host_model_on_grid=None if row.get('host_model') is None else np.asarray(row['host_model']),
            complexes=('mgii','hbeta_oiii','oii_nev_neiii_hgamma'))
        fit_seconds=perf_counter()-start
        for fitted in result.line_complexes.values():
            fitted.metadata.pop('_peak_parameter_state',None)
        with patch('qsospec.fitting.global_fit._solve_once_with_fallback',side_effect=AssertionError('Unexpected fit')), \
             patch('qsospec.fitting.global_fit.least_squares',side_effect=AssertionError('Unexpected optimizer')):
            start=perf_counter();statuses=recover_line_peaks(result);peak_seconds=perf_counter()-start
            start=perf_counter();diagnostic=qsospec.estimate_systemic_redshift(result);systemic_seconds=perf_counter()-start
        assert result.spectrum.z==row['z']
        report=dict(object_id=row['object_id'],fit_seconds=fit_seconds,peak_seconds=peak_seconds,
            systemic_seconds=systemic_seconds,recovery=statuses,diagnostic=diagnostic,
            spectral_optimizer_calls_during_postprocessing=0,input_redshift_unchanged=True)
        reports.append(report)
        print(row['object_id'],round(peak_seconds,3),round(systemic_seconds,3),diagnostic['status'],flush=True)
    Path(args.output).write_text(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
