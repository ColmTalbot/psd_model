#!/usr/bin/env python
"""

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import bilby
from bilby.core.prior import Uniform
from gwpy.timeseries import TimeSeries
from scipy.stats import binom, chi, norm
from gwosc.datasets import event_gps
from tqdm.auto import trange

from bilby_psd.models import SplineLorentzianPSD
from bilby_psd.likelihood import PSDLikelihood
from bilby_psd.priors_defs import Discrete, MaximumPrior, VariableBeta


def plot_lorentzians_prior(ifo, outdir, label):
    fig, ax = plt.subplots(figsize=(12, 7.8))
    ax.loglog(
        ifo.frequency_array[ifo.frequency_mask],
        np.abs(ifo.frequency_domain_strain[ifo.frequency_mask])
    )
    ax.set_xlim(ifo.minimum_frequency, ifo.maximum_frequency)
    fig.savefig(f"{outdir}/{label}_data")
    plt.close(fig)


Nsplines = int(sys.argv[1])
Nlorentzians = int(sys.argv[2])
detector = sys.argv[3]

event = 'GW170814'
duration = 8
time_of_event = event_gps(event)
sampling_frequency = 16384

outdir = 'outdir_real_data'
label = f'{event}_{detector}_noise_splines_{Nsplines}_lorentzians_{Nlorentzians}'

post_trigger_duration = 2
analysis_start = time_of_event + post_trigger_duration - duration

# Use gwpy to fetch the open data
analysis_data = TimeSeries.fetch_open_data(
    detector, analysis_start, analysis_start + duration,
    sample_rate=16384, cache=True
)

ifo = bilby.gw.detector.get_empty_interferometer(detector)
ifo.set_strain_data_from_gwpy_timeseries(analysis_data)
ifo.minimum_frequency = 20
ifo.maximum_frequency = 2**11

# Frequency bands with bad data taken from a mixture of by eye checks
# and https://www.gw-openscience.org/o2speclines/
bad_bands = dict(
    L1=[
        (46.05, 46.2),
        (59.75, 60.25),
        (305, 308),
        (314.5, 315.5),
        (331, 332),
        (495, 520),
        (612, 613),
        (615, 615.5),
        (629.5, 631.5),
        (918.5, 919),
        (922.5, 923),
        (944, 946),
        (998.5, 999),
        (1010, 1027),
        (1083, 1083.25),
        (1225, 1225.5),
        (1230.25, 1230.75),
        (1457, 1458),
        (1470, 1471.5),
        (1490, 1512),
        (1922, 1923),
        (1940, 1942),
        (1953.5, 1954.5),
        (1958, 1959),
        (1962, 1969),
        (1973, 1974),
        (1977.5, 1978.5),
        (1982, 1983),
        (1985, 1990),
    ],
    H1=[
        (46.05, 46.2),
        (59.75, 60.25),
        (298, 305),
        (498, 512),
        (595, 610),
        (985, 1015),
        (1450, 1490),
        (1920, 1970)
    ]
)
for band in bad_bands[detector]:
    _mask = (ifo.frequency_array >= band[0]) & (ifo.frequency_array <= band[1])
    ifo.strain_data.frequency_mask[_mask] = False

fixed_spline_points = np.logspace(
    np.log10(ifo.minimum_frequency), np.log10(ifo.maximum_frequency), Nsplines
)
midpoints = (fixed_spline_points[1:] * fixed_spline_points[:-1]) ** 0.5
priors = bilby.core.prior.ConditionalPriorDict()
for ii in range(Nsplines):
    priors[f'{ifo.name}_spline_frequency_{ii}'] = fixed_spline_points[ii]
    if ii == 0:
        low_end = ifo.minimum_frequency
    else:
        low_end = midpoints[ii - 1]
    if ii == Nsplines - 1:
        high_end = ifo.maximum_frequency
    else:
        high_end = midpoints[ii]
    temp_mask = (ifo.frequency_array >= low_end) & (ifo.frequency_array <= high_end)
    values = abs(ifo.frequency_domain_strain[temp_mask]) ** 2
    values = values[values > 0]

    key = f'{ifo.name}_spline_amplitude_{ii}'
    latex = f"SA{ii}"
    priors[key] = Uniform(
        np.log10(np.min(values)),
        np.log10(np.max(values)),
        name=key,
        latex_label=latex
    )

n_overlap = 2
plot_lorentzians_prior(ifo, outdir, label)

if Nlorentzians > 0:
    priors[f"{ifo.name}_n_lorentzians"] = Discrete(
        minimum=0, maximum=Nlorentzians, step_size=1,
        latex_label="$n_{L}$"
    )

for ii in range(Nlorentzians):

    key = f"{ifo.name}_lorentzian_frequency_{ii}"
    latex = f"LF{ii}"
    priors[key] = Uniform(
        minimum=ifo.minimum_frequency,
        maximum=ifo.maximum_frequency,
        name=key,
        latex_label=latex
    )

    key = f"{ifo.name}_lorentzian_amplitude_{ii}"
    latex = f"LA{ii}"
    if ii == 0:
        priors[key] = VariableBeta(
            minimum=0,
            maximum=4,
            alpha_parameter=f"{detector}_n_lorentzians",
            name=key,
            latex_label=latex
        )
    else:
        priors[key] = MaximumPrior(
            order=ii,
            minimum=0,
            maximum=4,
            name=key,
            latex_label=latex
        )

    key = f"{ifo.name}_lorentzian_quality_{ii}"
    latex = f"LQ{ii}"
    priors[key] = Uniform(-2, 2, key, latex_label=latex)

psd = SplineLorentzianPSD(
    f'{ifo.name}',
    ifo.frequency_array[ifo.frequency_mask],
    parameters=priors.sample()
)

ifo.power_spectral_density = psd
ifos = bilby.gw.detector.InterferometerList([ifo])
likelihood = PSDLikelihood(ifo=ifo, coarsen=8)

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', nlive=1000,
    dlogz=0.01, nthreads=1, walks=10, nact=1, ntemps=10, queue_size=3,
    outdir=outdir, label=label, evidence_tolerance=5, multimodal=True)

result.priors.sample()
result.save_to_file(overwrite=True)

if Nlorentzians > 0:
    result.plot_corner(
        parameters=[f"{detector}_n_lorentzians"],
        filename=f"{outdir}/{label}_n_lorentzians_corner"
    )

full_posterior = result.posterior.copy()
for ll in np.arange(Nlorentzians):
    result.posterior = result.posterior[
        result.posterior[f"{detector}_n_lorentzians"] >= ll + 1
    ]
    if len(result.posterior) <= len(priors.keys()):
        break
    keys = [
        f"{detector}_lorentzian_{parameter}_{ll}"
        for parameter in ["frequency", "amplitude", "quality"]
    ]
    result.plot_corner(keys, filename=f"{outdir}/{label}_lorentzian_{ll + 1}")
result.posterior = full_posterior

if Nsplines > 0:
    spline_keys = [key for key in priors.keys() if "spline_amplitude" in key]
    result.plot_corner(spline_keys, filename=f"{outdir}/{label}_spline_corner")

freq = ifo.frequency_array[ifo.frequency_mask]
data = ifo.strain_data.frequency_domain_strain[ifo.frequency_mask]

for ii in trange(min(100, len(result.posterior))):
    ifo.power_spectral_density.parameters.update(result.posterior.iloc[ii])
    try:
        psds = np.vstack([psds, ifo.power_spectral_density_array[ifo.frequency_mask]])
    except:
        psds = ifo.power_spectral_density_array[ifo.frequency_mask]

fig, ax = plt.subplots(figsize=(12, 7.8))
ax.loglog(freq, np.abs(data) * (2 / duration) ** 0.5, color='C0', zorder=0, label="Data")

plt.loglog(freq, np.median(psds, axis=0)**0.5, color='C1',
           zorder=2, label="Median")
plt.loglog(freq, np.mean(psds, axis=0)**0.5, color='C3',
           zorder=2, label="Mean")
plt.fill_between(freq, np.quantile(psds, 0.05, axis=0)**0.5,
                 np.quantile(psds, 0.95, axis=0)**0.5, alpha=0.8, color='C2',
                 zorder=1, label="90% C.I.")
ax.legend()
ax.set_xscale("linear")
ax.set_xlim(ifo.minimum_frequency, ifo.maximum_frequency)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude Specral Density [Hz^{-1/2}]")

plt.savefig(f"{result.outdir}/{result.label}_check")
plt.close()

plt.figure(figsize=(8, 5))
for method, colour, dist, _label in zip(
    [np.real, np.imag, np.abs],
    ["C0", "C1", "C2"],
    [norm(), norm(), chi(df=2)],
    ["Real", "Imaginary", "Magnitude"]
):
    for ii in trange(100):
        plt.plot(
            np.linspace(0, 1, len(data)),
            dist.cdf(np.sort(method(data) / psds[ii] ** 0.5 * (4 / duration) ** 0.5))
            - np.linspace(0, 1, len(data)),
            color=colour, alpha=0.03
        )
    plt.plot(
        np.linspace(0, 1, len(data)),
        dist.cdf(np.sort(method(data) / np.mean(psds, axis=0) ** 0.5 * (4 / duration) ** 0.5))
        - np.linspace(0, 1, len(data)),
        color=colour,
        alpha=1,
        label=_label
    )
ax = plt.gca()
x_values = np.linspace(0, 1, 1000)
confidence_interval = [0.68, 0.95, 0.997]
confidence_interval_alpha = [0.1] * 3
N = len(data)
for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
    edge_of_bound = (1. - ci) / 2.
    lower = binom.ppf(1 - edge_of_bound, N, x_values) / N
    upper = binom.ppf(edge_of_bound, N, x_values) / N
    lower[0] = 0
    upper[0] = 0
    ax.fill_between(x_values, lower - x_values, upper - x_values, alpha=alpha, color='k')
plt.xlabel("Expected CDF")
plt.ylabel("Sample CDF - Expected CDF")
plt.xlim(0, 1)
plt.legend(loc="upper left")
plt.savefig(f"{outdir}/{label}_ppp.png")
plt.close()
