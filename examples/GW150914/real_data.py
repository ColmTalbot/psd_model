#!/usr/bin/env python
"""

"""
import argparse

import numpy as np
import bilby
from bilby.core.prior import Uniform, TruncatedNormal
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

from bilby_psd.models import SplineLorentzianPSD
from bilby_psd.likelihood import PSDLikelihood
from bilby_psd.priors_defs import SpikeAndSlab, MinimumPrior


parser = argparse.ArgumentParser()
parser.add_argument('--trigger-time', default=1126259462.4)
parser.add_argument('--duration', default=4, type=int)
parser.add_argument('-s', '--nsplines', type=int, required=True)
parser.add_argument('-l', '--nlorentzians', type=int, required=True)
parser.add_argument('-d', '--detector', type=str, required=True)
parser.add_argument('--max-frequency', type=int, default=512)
parser.add_argument('--min-frequency', type=int, default=15)
parser.add_argument('--min-lorentzian-frequency', type=int, default=20)
parser.add_argument('--max-lorentzian-frequency', type=int, default=550)
parser.add_argument('--sampler', type=str, default="dynesty")
parser.add_argument('--nlive', type=int, default=500)
parser.add_argument('--cpus', type=int, default=1)
args = parser.parse_args()

outdir = f'{args.detector}_{args.max_frequency}'
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
label = f'noise_splines{args.nsplines}_lorentzians{args.nlorentzians}'

post_trigger_duration = 2
analysis_start = args.trigger_time + post_trigger_duration - args.duration

# Use gwpy to fetch the open data
analysis_data = TimeSeries.fetch_open_data(
    args.detector, analysis_start, analysis_start + args.duration,
    sample_rate=4096, cache=True)

ifo = bilby.gw.detector.get_empty_interferometer(args.detector)
ifo.set_strain_data_from_gwpy_timeseries(analysis_data)
ifo.minimum_frequency = args.min_frequency
ifo.maximum_frequency = args.max_frequency

# Set up spline priors
fixed_spline_points = np.logspace(
    np.log10(ifo.minimum_frequency),
    np.log10(ifo.maximum_frequency),
    args.nsplines)

priors = bilby.core.prior.PriorDict()
for ii in range(args.nsplines):
    priors[f'{ifo.name}_spline_frequency_{ii}'] = fixed_spline_points[ii]
    key = f'{ifo.name}_spline_amplitude_{ii}'
    latex = f"SA{ii}"
    priors[key] = Uniform(-57, -35, key, latex_label=latex)

# Set up lorentzian priors
for ii in range(args.nlorentzians):
    key = f"{ifo.name}_lorentzian_frequency_{ii}"
    latex = f"LF{ii}"
    if ii > 0:
        priors[key] = MinimumPrior(
            order=args.nlorentzians - ii,
            duration=args.duration,
            minimum=args.min_lorentzian_frequency,
            maximum=args.max_lorentzian_frequency,
            name=key,
            latex_label=latex
        )
    else:
        priors[key] = bilby.core.prior.Beta(
            minimum=args.min_lorentzian_frequency,
            maximum=ifo.maximum_frequency,
            alpha=1,
            beta=args.nlorentzians,
            name=key,
            latex_label=latex
        )

    key = f"{ifo.name}_lorentzian_amplitude_{ii}"
    latex = f"LA{ii}"
    priors[key] = SpikeAndSlab(
        mix=0.5,
        slab=Uniform(-50, -35),
        name=key,
        latex_label=latex)

    key = f"{ifo.name}_lorentzian_quality_{ii}"
    latex = f"LQ{ii}"
    priors[key] = TruncatedNormal(
        mu=-2, sigma=1, minimum=-2, maximum=1, name=key, latex_label=latex)

farray = np.linspace(ifo.minimum_frequency, ifo.maximum_frequency, 101)
psd = SplineLorentzianPSD(
    f'{ifo.name}', farray, parameters=priors.sample())

ifo.power_spectral_density = psd
likelihood = PSDLikelihood(ifo=ifo)

if args.sampler == 'dynesty':
    sampler_kwargs = dict(
        dlogz=.5, queue_size=args.cpus, n_effective=1000, walks=50, nact=2)
elif args.sampler == 'ptemcee':
    sampler_kwargs = dict(nsamples=1000, ntemps=3, nwalkers=100, nthreads=args.cpus, pos0='minimize')
else:
    sampler_kwargs = dict()

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, outdir=outdir, label=label,
    sampler=args.sampler, nlive=args.nlive, **sampler_kwargs)

if args.nlorentzians > 0:
    lorentzian_keys = [key for key in priors.keys() if "lorentzian_frequency" in key]
    result.plot_corner(lorentzian_keys, filename=f"{outdir}/{label}_lorentz_freq_corner")

    lorentzian_keys = [key for key in priors.keys() if "lorentzian_amplitude" in key]
    result.plot_corner(lorentzian_keys, filename=f"{outdir}/{label}_lorentz_amp_corner")

    lorentzian_keys = [key for key in priors.keys() if "lorentzian_quality" in key]
    result.plot_corner(lorentzian_keys, filename=f"{outdir}/{label}_lorentz_q_corner")

if args.nsplines > 0:
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
                 zorder=1, label=r"90\% C.I.")
ax.legend()
ax.set_xlim(ifo.minimum_frequency, ifo.maximum_frequency)
ax.set_xlabel("Frequency [Hz]")

plt.savefig(f"{result.outdir}/{result.label}_check")
