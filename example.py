#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""
from __future__ import division, print_function

import os

import numpy as np
import matplotlib.pyplot as plt
import bilby

from models import SplineLorentzianPSD
from likelihood import PSDLikelihood, PSDGravitationalWaveTransient


# Specify the output directory and the name of the simulation.
outdir = 'outdir'
label = 'noise'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

duration = 4
sampling_frequency = 2048

injection_parameters = dict(
    mass_1=36., mass_2=29., a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=5000., iota=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

# Fixed arguments passed into the source model
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=50., minimum_frequency=20.)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

# ifo = bilby.gw.detector.get_empty_interferometer('CE')
ifos = bilby.gw.detector.InterferometerList(['H1'])
ifos.set_strain_data_from_power_spectral_densities(
    duration=duration, sampling_frequency=sampling_frequency)
ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)
ifo = ifos[0]

# parameters = {'{}_spline_frequency_{}'.format(ifo.name, ii): ff
#               for ii, ff in enumerate(np.logspace(1, np.log10(2048), 11))}
# parameters.update({'{}_spline_amplitude_{}'.format(ifo.name, ii): - 40
#                    for ii in range(11)})

psd_params = {
    '{}_spline_amplitude_{}'.format(ifo.name, ii):
        np.log10(
            ifo.power_spectral_density.power_spectral_density_interpolated(
                np.logspace(np.log10(ifo.minimum_frequency),
                            np.log10(ifo.maximum_frequency), 11)[ii]))
    for ii in range(11)}
psd_params.update({
    '{}_spline_frequency_{}'.format(ifo.name, ii):
        np.logspace(np.log10(ifo.minimum_frequency),
                    np.log10(ifo.maximum_frequency), 11)[ii]
    for ii in range(11)})

injection_parameters.update(psd_params)

psd = SplineLorentzianPSD(
    '{}'.format(ifo.name), np.linspace(10, 2048, 1001), parameters=psd_params)

ifo.power_spectral_density = psd

like = PSDLikelihood(ifo=ifo)

like.parameters.update(psd_params)

priors = bilby.core.prior.PriorDict()

priors.update(psd_params)

for key in psd_params:
    if 'amplitude' in key:
        priors[key] = bilby.core.prior.Uniform(
            psd_params[key] - 0.3, psd_params[key] + 0.3,
            latex_label='${}$'.format(' '.join(key.split())))

# import IPython; IPython.embed()

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=like, priors=priors, sampler='pymultinest', npoints=500,
    injection_parameters=psd_params, outdir=outdir, label=label)

# for ii in range(len(result.posterior)):
for ii in range(200):
    ifo.power_spectral_density.parameters.update(result.posterior.iloc[ii])
    try:
        psds = np.vstack([psds, ifo.power_spectral_density_array])
    except:
        psds = ifo.power_spectral_density_array

# Make a corner plot.
result.plot_corner()

priors.update(bilby.gw.prior.BBHPriorDict())
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 1,
    maximum=injection_parameters['geocent_time'] + 1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')
for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
            'dec', 'geocent_time', 'phase', 'mass_2', 'iota', 'mass_1']:
    priors[key] = injection_parameters[key]

likelihood = PSDGravitationalWaveTransient(
    ifos=ifos, wfg=waveform_generator)

label = 'signal_noise'

# Run sampler.  In this case we're going to use the `dynesty` sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=100,
    injection_parameters=injection_parameters, outdir=outdir, label=label, use_ratio=False)

plt.figure(figsize=(12, 7.8))
ifo.power_spectral_density.parameters.update(psd_params)
plt.loglog(ifo.frequency_array, np.median(psds, axis=0))
plt.fill_between(ifo.frequency_array, np.quantile(psds, 0.05, axis=0),
                 np.quantile(psds, 0.95, axis=0), alpha=0.3)
new_ifo = bilby.gw.detector.get_empty_interferometer('CE')
new_ifo.set_strain_data_from_power_spectral_density(
    duration=4, sampling_frequency=2048)
plt.loglog(ifo.frequency_array, new_ifo.power_spectral_density_array)
plt.xlim(ifo.minimum_frequency, ifo.maximum_frequency)
plt.savefig(os.path.join(outdir, 'samples.png'))

# Make a corner plot.
result.plot_corner(parameters=['luminosity_distance'])
