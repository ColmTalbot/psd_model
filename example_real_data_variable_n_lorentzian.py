#!/usr/bin/env python
"""

"""
from __future__ import division, print_function
import sys

import tqdm
import numpy as np
import bilby
import matplotlib.pyplot as plt
import peakutils

from models import Discrete, SplineLorentzianPSD
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

event = 'GW170814'
trigger_time = bilby.gw.utils.get_event_time(event)
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


# Specify the output directory and the name of the simulation.
Nlorentzians = 8
label = f'{event}_splines{Nsplines}_variable_lorentzians{Nlorentzians}_dynesty'

psd = SplineLorentzianPSD(
    f'{ifo.name}', np.linspace(10, 2048, 1001), parameters=psd_params)

ifo.power_spectral_density = psd

like = PSDLikelihood(ifo=ifo)
like.parameters.update(psd_params)


def constrain_amplitudes(params):
    ordered = True
    old_amp = 0
    for ii in range(int(params['H1_n_lorentzians'])):
        if params[f'H1_lorentzian_amplitude_{ii}'] > old_amp:
            ordered = False
            break
        old_amp = params[f'H1_lorentzian_amplitude_{ii}']
    params['ordered'] = int(ordered)
    return params


priors = bilby.core.prior.PriorDict(conversion_function=constrain_amplitudes)

priors['oredered'] = bilby.core.prior.Constraint(minimum=0.1, maximum=10)
priors['H1_n_lorentzians'] = Discrete(
    minimum=0, maximum=Nlorentzians - 1, step_size=1,
    latex_label='$N lorentzians$')

for ii in range(Nlorentzians):
    priors[f'H1_lorentzian_amplitude_{ii}'] = bilby.core.prior.Uniform(
        -50, -40, boundary='reflective',
        latex_label=f'H1 lorentzian amplitude {ii}')
    priors[f'H1_lorentzian_frequency_{ii}'] = bilby.core.prior.Uniform(
        10, 1024, f'H1_lorentzian_frequency_{ii}', boundary='reflective',
        latex_label=f'f_{ii}')
    priors[f'H1_lorentzian_quality_{ii}'] = bilby.core.prior.Uniform(
        -3, 2, f'H1_lorentzian_quality_{ii}', boundary='reflective',
        latex_label=f'q_{ii}')

for key in psd_params:
    if 'spline_amplitude' in key:
        priors[key] = bilby.core.prior.Uniform(
            -50, -40,
            latex_label='{}'.format(' '.join(key.split('_'))))

result = bilby.run_sampler(
    likelihood=like, priors=priors, sampler='dynesty', nlive=100,
    enlarge=1, walks=10,
    outdir=outdir, label=label)

result.plot_corner(parameters=['H1_n_lorentzians'])

for ii in tqdm.tqdm(range(200)):
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
                 zorder=1, label="90% C.I.")
ax.legend()
ax.set_xlim(20, sampling_frequency / 2.)
ax.set_xlabel("Frequency [Hz]")

plt.savefig(f"{result.outdir}/{result.label}_check")



