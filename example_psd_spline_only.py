#!/usr/bin/env python
"""

"""
from __future__ import division, print_function
import sys

import numpy as np
import bilby
import matplotlib.pyplot as plt

from models import SplineLorentzianPSD
from likelihood import PSDLikelihood

Nsplines = int(sys.argv[1])
outdir = 'psd_only'
label = f'noise_splines{Nsplines}'

duration = 4
sampling_frequency = 4096

np.random.seed(123)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=20., minimum_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    duration=duration, sampling_frequency=sampling_frequency)
ifo = ifos[0]

fixed_spline_points = np.logspace(np.log10(ifo.minimum_frequency),
                                  np.log10(ifo.maximum_frequency), Nsplines)
psd_params = dict()
for ii in range(Nsplines):
    psd_params[f'{ifo.name}_spline_frequency_{ii}'] = fixed_spline_points[ii]
    psd_params[f'{ifo.name}_spline_amplitude_{ii}'] = -45

psd = SplineLorentzianPSD(
    f'{ifo.name}', np.linspace(10, 2048, 1001), parameters=psd_params)

ifo.power_spectral_density = psd

like = PSDLikelihood(ifo=ifo)
like.parameters.update(psd_params)

priors = bilby.core.prior.PriorDict()
priors.update(psd_params)

for key in psd_params:
    if 'amplitude' in key:
        priors[key] = bilby.core.prior.Uniform(
            -50, -40,
            latex_label='{}'.format(' '.join(key.split('_'))))

result = bilby.run_sampler(
    likelihood=like, priors=priors, sampler='nestle', nlive=500,
    outdir=outdir, label=label)

result.plot_corner()

for ii in range(200):
    ifo.power_spectral_density.parameters.update(result.posterior.iloc[ii])
    try:
        psds = np.vstack([psds, ifo.power_spectral_density_array])
    except:
        psds = ifo.power_spectral_density_array

fig, ax = plt.subplots(figsize=(12, 7.8))
ax.loglog(ifo.frequency_array, np.abs(ifo.strain_data.frequency_domain_strain),
          color='C0', zorder=0, label="Data")

plt.loglog(ifo.frequency_array, np.median(psds, axis=0)**0.5, color='C1',
           zorder=2, label="Median")
plt.fill_between(ifo.frequency_array, np.quantile(psds, 0.05, axis=0)**0.5,
                 np.quantile(psds, 0.95, axis=0)**0.5, alpha=0.8, color='C2',
                 zorder=1, label="90\% C.I.")
ax.legend()
ax.set_xlim(20, sampling_frequency / 2.)
ax.set_xlabel("Frequency [Hz]")

plt.savefig(f"{result.outdir}/{result.label}_check")
