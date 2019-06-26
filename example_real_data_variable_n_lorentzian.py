#!/usr/bin/env python
"""

"""
from __future__ import division, print_function

import tqdm
import numpy as np
import bilby
import matplotlib.pyplot as plt

from models import Discrete, SplineLorentzianPSD
from likelihood import PSDLikelihood

outdir = 'variable_outdir'
Nsplines = 4
Nlorentzians = 5

event = 'GW150914'
trigger_time = bilby.gw.utils.get_event_time(event) - 8
duration = 4
start_time = trigger_time + 2 - duration
sampling_frequency = 4096
detector = 'H1'

np.random.seed(123)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=20., minimum_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

strain = bilby.gw.detector.InterferometerStrainData(
    minimum_frequency=20, maximum_frequency=500, roll_off=0.4)
strain.set_from_open_data(
    name=detector, start_time=start_time, duration=duration,
    outdir=outdir, cache=True)
ifo = bilby.gw.detector.get_empty_interferometer(detector)
ifo.strain_data = strain
ifo.plot_data(outdir=outdir)

fixed_spline_points = np.logspace(np.log10(ifo.minimum_frequency),
                                  np.log10(ifo.maximum_frequency), Nsplines)
print("Fixed spine points=", fixed_spline_points)

psd_params = dict()
for ii in range(Nsplines):
    psd_params[f'{ifo.name}_spline_frequency_{ii}'] = fixed_spline_points[ii]
    psd_params[f'{ifo.name}_spline_amplitude_{ii}'] = -45

# Specify the output directory and the name of the simulation.
label = f'{event}_splines{Nsplines}_variable_lorentzians{Nlorentzians}_dynesty'

psd = SplineLorentzianPSD(
    f'{ifo.name}', np.linspace(20, 500, 1001), parameters=psd_params)

ifo.power_spectral_density = psd

like = PSDLikelihood(ifo=ifo)
like.parameters.update(psd_params)


def constrain_amplitudes(params):
    ordered = True
    old_amp = 0
    if 'H1_n_lorentzians' in params:
        for ii in range(int(params['H1_n_lorentzians'])):
            if params[f'H1_lorentzian_amplitude_{ii}'] > old_amp:
                ordered = False
                break
            old_amp = params[f'H1_lorentzian_amplitude_{ii}']
    params['ordered'] = int(ordered)
    return params


priors = bilby.core.prior.PriorDict(conversion_function=constrain_amplitudes)

priors['ordered'] = bilby.core.prior.Constraint(minimum=0.1, maximum=10)
if Nlorentzians == 0:
    priors['H1_n_lorentzians'] = 0
else:
    priors['H1_n_lorentzians'] = Discrete(
        minimum=1, maximum=Nlorentzians, step_size=1,
        latex_label='$N lorentzians$')

for ii in range(Nlorentzians):
    priors[f'H1_lorentzian_amplitude_{ii}'] = bilby.core.prior.Uniform(
        -47, -30, boundary='reflective',
        latex_label=f'H1 lorentzian amplitude {ii}')
    priors[f'H1_lorentzian_frequency_{ii}'] = bilby.core.prior.Uniform(
        20, 400, f'H1_lorentzian_frequency_{ii}', boundary='reflective',
        latex_label=f'f_{ii}')
    priors[f'H1_lorentzian_quality_{ii}'] = bilby.core.prior.Uniform(
        -10, 10, f'H1_lorentzian_quality_{ii}', boundary='reflective',
        latex_label=f'q_{ii}')

for key in psd_params:
    if 'spline_amplitude' in key:
        priors[key] = bilby.core.prior.Uniform(
            -548, -42,
            latex_label='{}'.format(' '.join(key.split('_'))))

result = bilby.run_sampler(
    likelihood=like, priors=priors, sampler='dynesty', nlive=500,
    walks=20, check_point=True, n_check_point=5000,
    outdir=outdir, label=label)

result.plot_corner(
    parameters=(['H1_n_lorentzians'] +
                [f'H1_lorentzian_frequency_{i}' for i in range(Nlorentzians)] +
                [f'H1_lorentzian_quality_{i}' for i in range(Nlorentzians)]))

nsamples = 200
psds = np.zeros((nsamples, np.sum(ifo.frequency_mask)))
for ii in tqdm.tqdm(range(nsamples)):
    ifo.power_spectral_density.parameters.update(result.posterior.iloc[ii])
    psds[ii] = ifo.power_spectral_density_array[ifo.frequency_mask]

fig, ax = plt.subplots(figsize=(12, 7.8))
ax.loglog(ifo.frequency_array[ifo.frequency_mask],
          np.abs(ifo.strain_data.frequency_domain_strain[ifo.frequency_mask]),
          color='C0', zorder=0, label="Data")

plt.loglog(ifo.frequency_array[ifo.frequency_mask],
           np.median(psds, axis=0)**0.5, color='C1',
           zorder=2, label="Median")
plt.fill_between(ifo.frequency_array[ifo.frequency_mask], np.quantile(psds, 0.05, axis=0)**0.5,
                 np.quantile(psds, 0.95, axis=0)**0.5, alpha=0.8, color='C2',
                 zorder=1, label="90% C.I.")
ax.set_ylim(1e-25, 5e-20)
ax.legend()
ax.set_xlim(20, ifo.maximum_frequency)
ax.set_xlabel("Frequency [Hz]")

plt.savefig(f"{result.outdir}/{result.label}_check")
