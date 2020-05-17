#!/usr/bin/env python
"""

"""
import sys

import numpy as np
import bilby
from bilby.core.prior import Uniform
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

from bilby_psd.models import SplineLorentzianPSD
from bilby_psd.likelihood import PSDLikelihood
from bilby_psd.priors_defs import SpikeAndSlab

Nsplines = int(sys.argv[1])
Nlorentzians = int(sys.argv[2])

outdir = 'outdir_real_data'
label = f'noise_splines{Nsplines}_lorentzians{Nlorentzians}'

event = 'GW150914'
time_of_event = 1126259462.4
duration = 4
sampling_frequency = 4096

post_trigger_duration = 2
analysis_start = time_of_event + post_trigger_duration - duration

detector = "H1"

# Use gwpy to fetch the open data
analysis_data = TimeSeries.fetch_open_data(
    detector, analysis_start, analysis_start + duration, sample_rate=4096,
    cache=True)

ifo = bilby.gw.detector.get_empty_interferometer(detector)
ifo.set_strain_data_from_gwpy_timeseries(analysis_data)
ifo.minimum_frequency = 20
ifo.maximum_frequency = 2**9

fixed_spline_points = np.logspace(np.log10(ifo.minimum_frequency),
                                  np.log10(ifo.maximum_frequency), Nsplines)
priors = bilby.core.prior.PriorDict()
for ii in range(Nsplines):
    priors[f'{ifo.name}_spline_frequency_{ii}'] = fixed_spline_points[ii]

    key = f'{ifo.name}_spline_amplitude_{ii}'
    latex = f"SA{ii}"
    priors[key] = Uniform(-50, -40, key, latex_label=latex)

lorentzians_frequency_bins = np.logspace(
    np.log10(ifo.minimum_frequency),
    np.log10(ifo.maximum_frequency),
    1 + Nlorentzians)

for ii in range(Nlorentzians):

    key = f"{ifo.name}_lorentzian_frequency_{ii}"
    latex = f"LF{ii}"
    # priors[key] = Uniform(
    #    lorentzians_frequency_bins[ii],
    #    lorentzians_frequency_bins[ii + 1], key, latex_label=latex)
    priors[key] = Uniform(
        ifo.minimum_frequency,
        ifo.maximum_frequency,
        key, latex_label=latex)

    key = f"{ifo.name}_lorentzian_amplitude_{ii}"
    latex = f"LA{ii}"
    priors[key] = SpikeAndSlab(
        mix=0.5,
        slab=Uniform(-60, -40),
        name=key,
        latex_label=latex)

    key = f"{ifo.name}_lorentzian_quality_{ii}"
    latex = f"LQ{ii}"
    priors[key] = Uniform(
        -2, 1, key, latex_label=latex)

farray = np.linspace(ifo.minimum_frequency, ifo.maximum_frequency, 501)
psd = SplineLorentzianPSD(
    f'{ifo.name}', farray, parameters=priors.sample())

ifo.power_spectral_density = psd
likelihood = PSDLikelihood(ifo=ifo)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='pymultinest', nlive=250,
    outdir=outdir, label=label, evidence_tolerance=5, multimodal=True)

if Nlorentzians > 0:
    lorentzian_keys = [key for key in priors.keys() if "lorentzian_frequency" in key]
    result.plot_corner(lorentzian_keys, filename=f"{outdir}/{label}_lorentz_freq_corner")

    lorentzian_keys = [key for key in priors.keys() if "lorentzian_amplitude" in key]
    result.plot_corner(lorentzian_keys, filename=f"{outdir}/{label}_lorentz_amp_corner")

    lorentzian_keys = [key for key in priors.keys() if "lorentzian_quality" in key]
    result.plot_corner(lorentzian_keys, filename=f"{outdir}/{label}_lorentz_q_corner")

if Nsplines > 0:
    spline_keys = [key for key in priors.keys() if "spline_amplitude" in key]
    result.plot_corner(spline_keys, filename=f"{outdir}/{label}_spline_corner")

freq = ifo.frequency_array[ifo.frequency_mask]
data = ifo.strain_data.frequency_domain_strain[ifo.frequency_mask]

for ii in range(min(100, len(result.posterior))):
    ifo.power_spectral_density.parameters.update(result.posterior.iloc[ii])
    try:
        psds = np.vstack([psds, ifo.power_spectral_density_array[ifo.frequency_mask]])
    except:
        psds = ifo.power_spectral_density_array[ifo.frequency_mask]

fig, ax = plt.subplots(figsize=(12, 7.8))
ax.loglog(freq, np.abs(data), color='C0', zorder=0, label="Data")

plt.loglog(freq, np.median(psds, axis=0)**0.5, color='C1',
           zorder=2, label="Median")
plt.fill_between(freq, np.quantile(psds, 0.05, axis=0)**0.5,
                 np.quantile(psds, 0.95, axis=0)**0.5, alpha=0.8, color='C2',
                 zorder=1, label="90\% C.I.")
ax.legend()
ax.set_xlim(ifo.minimum_frequency, ifo.maximum_frequency)
ax.set_xlabel("Frequency [Hz]")

plt.savefig(f"{result.outdir}/{result.label}_check")
