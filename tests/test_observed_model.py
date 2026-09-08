from types import SimpleNamespace
import numpy as np
import qsospec


def test_observed_total_model_host_frame_and_overlap():
    wave=np.linspace(3380,5100,1500)
    s=qsospec.Spectrum.from_arrays(wave,np.ones(len(wave))*8,err=np.ones(len(wave)),z=1.,wave_frame='rest',flux_unit='cgs',flux_scale=1.e-17)
    continuum=SimpleNamespace(model=np.ones(len(wave))*4,success=True)
    def fit(v):return SimpleNamespace(model=np.ones(len(wave))*v,wave_rest=wave,metrics={},metric_errors={},warnings=[],success=True)
    r=SimpleNamespace(spectrum=s,total_spectrum=s,continuum=continuum,host_model_on_quasar_grid=np.ones(len(wave))*2,
                      line_complexes={'oii_nev_neiii_hgamma':fit(1),'hbeta_oiii':fit(3)})
    a=qsospec.reconstruct_observed_model(r)
    assert np.allclose(a['flux'],4.e-17)
    assert np.allclose(a['model'][(wave>4700)&(wave<4900)],4.5e-17,rtol=1e-10,atol=0)
    r.line_complexes=dict(reversed(list(r.line_complexes.items())))
    b=qsospec.reconstruct_observed_model(r)
    np.testing.assert_array_equal(a['model'],b['model'])
    r.line_complexes['hbeta_oiii'].success=False
    assert not qsospec.reconstruct_observed_model(r)['supported'][(wave>4700)&(wave<4900)].any()
