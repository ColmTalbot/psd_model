#!/usr/bin/env python
"""

"""
from __future__ import division, print_function
import sys

import numpy as np
import bilby
import matplotlib.pyplot as plt
import peakutils

from models import SplineLorentzianPSD
from likelihood import PSDLikelihood

outdir = 'data_psd'
Nsplines = 7

duration = 4
sampling_frequency = 2048

np.random.seed(123)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=20., minimum_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

GW150914 = 1126259462
trigger_time = GW150914 - 20
ifo = bilby.gw.detector.get_interferometer_with_open_data(
    'H1', trigger_time=trigger_time)
ifo.plot_data(outdir=outdir)
plt.show()
fixed_spline_points = np.logspace(np.log10(ifo.minimum_frequency),
                                  np.log10(ifo.maximum_frequency), Nsplines)
print(fixed_spline_points)
psd_params = dict()
for ii in range(Nsplines):
    psd_params[f'{ifo.name}_spline_frequency_{ii}'] = fixed_spline_points[ii]
    psd_params[f'{ifo.name}_spline_amplitude_{ii}'] = -45

if 'check' in sys.argv:
    idxs = peakutils.indexes(np.abs(ifo.strain_data.frequency_domain_strain), thres=0.5,
                             min_dist=2)
    print(ifo.frequency_array[idxs])
    plt.loglog(ifo.frequency_array, np.abs(ifo.frequency_domain_strain))
    for f in ifo.frequency_array[idxs]:
        plt.axvline(f, color='C1')
    plt.show()
    sys.exit()

#print(fixed_lorenztians)
#for ii in range(len(fixed_lorenztians)):
#    psd_params[f"{ifo.name}_lorentzian_frequency_{ii}"] = fixed_lorenztians[ii]
#    psd_params[f'{ifo.name}_lorentzian_amplitude_{ii}'] = -45
#    psd_params[f'{ifo.name}_lorentzian_quality_{ii}'] = 2
#

fixed_lorentzians_frequencies = [36, 40.75, 60, 120, 180, 332, 502, 991.75]
fixed_lorentzians_amplitude = [-42] * len(fixed_lorentzians_frequencies)
fixed_lorentzians_qualitiy = [-1] * len(fixed_lorentzians_frequencies)

# Specify the output directory and the name of the simulation.
Nlorentzians = len(fixed_lorentzians_frequencies)
label = f'GW150914_splines{Nsplines}_lorentzians{Nlorentzians}'


for ii in range(Nlorentzians):
    psd_params[f"{ifo.name}_lorentzian_frequency_{ii}"] = fixed_lorentzians_frequencies[ii]
    psd_params[f'{ifo.name}_lorentzian_amplitude_{ii}'] = fixed_lorentzians_amplitude[ii]
    psd_params[f'{ifo.name}_lorentzian_quality_{ii}'] = fixed_lorentzians_qualitiy[ii]

psd = SplineLorentzianPSD(
    f'{ifo.name}', np.linspace(10, 2048, 1001), parameters=psd_params)

ifo.power_spectral_density = psd

like = PSDLikelihood(ifo=ifo)
like.parameters.update(psd_params)

priors = bilby.core.prior.PriorDict()
priors.update(psd_params)

for key in psd_params:
    if 'spline_amplitude' in key:
        priors[key] = bilby.core.prior.Uniform(
            -50, -40,
            latex_label='{}'.format(' '.join(key.split('_'))))
    if 'quality' in key:
        priors[key] = bilby.core.prior.Uniform(
            -5, 2,
            latex_label='{}'.format(' '.join(key.split('_'))))
    if 'lorentzian_amplitude' in key:
        priors[key] = bilby.core.prior.Uniform(
            -50, -20,
            latex_label='{}'.format(' '.join(key.split('_'))))
    if 'lorentzian_frequency' in key:
        priors[key] = bilby.core.prior.Uniform(
            psd_params[key] - 0.5, psd_params[key] + 0.5,
            latex_label='{}'.format(' '.join(key.split('_'))))

result = bilby.run_sampler(
    likelihood=like, priors=priors, sampler='dynesty', nlive=200,
    enlarge=1, walks=10,
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



