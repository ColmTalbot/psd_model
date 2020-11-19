#!/usr/bin/env python
import configargparse

import numpy as np
import bilby
from bilby.core.prior import Uniform, PowerLaw
from bilby_pipe.utils import convert_string_to_dict
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.signal import find_peaks

from bilby_psd.models import SplineLorentzianPSD
from bilby_psd.lines import LineList
from bilby_psd.likelihood import PSDLikelihood
from bilby_psd.priors_defs import SpikeAndSlab, MinimumPrior


def get_gamma_prior(args, key, ii):
    latex_label = f"LQ{ii}"
    return PowerLaw(
        alpha=0,
        minimum=args.gamma_min,
        maximum=args.gamma_max,
        name=key,
        latex_label=latex_label)


def get_amplitude_prior(args, key, ii):
    latex = f"LA{ii}"
    return SpikeAndSlab(
        mix=args.lorentzian_mix,
        slab=Uniform(args.lorentzian_min, args.lorentzian_max),
        name=key,
        latex_label=latex)


def plot_data_and_psd_prior(ifo, priors, outdir, label):
    fig, ax = plt.subplots()

    df = ifo.strain_data.frequency_array[1] - ifo.strain_data.frequency_array[0]
    asd = bilby.gw.utils.asd_from_freq_series(
        freq_data=ifo.strain_data.frequency_domain_strain, df=df)

    freq = ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask]
    ax.loglog(freq, asd[ifo.strain_data.frequency_mask],
              color='C0', label=ifo.name)

    N = 1000
    prior_draws = np.zeros((N, len(freq)))
    for ii in range(N):
        ifo.power_spectral_density.parameters.update(priors.sample())
        prior_draws[ii] = ifo.amplitude_spectral_density_array[ifo.strain_data.frequency_mask]
        ax.loglog(freq, prior_draws[ii], color='C1', lw=0.5, alpha=0.1, zorder=-100)

    ax.grid(True)
    ax.set_ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
    ax.set_xlabel(r'Frequency [Hz]')
    fig.tight_layout()
    fig.savefig(f"{outdir}/{label}_data_and_prior")


def main():
    parser = configargparse.ArgParser()
    parser.add('config', is_config_file=True, help='config file path')
    parser.add_argument('--trigger-time', type=float, required=True)
    parser.add_argument('--duration', default=4, type=int)
    parser.add_argument('-s', '--nsplines', type=int, required=True)
    parser.add_argument('-l', '--nlorentzians', type=int, required=True)
    parser.add_argument('--fixed-lorentzians-file', type=str, default=None)
    parser.add_argument('--fixed-lorentzians', action="append", type=float, default=[])
    parser.add_argument('--fixed-lorentzians-uncertainty', type=float, default=0.5)
    parser.add_argument('--find-peaks', action="store_true")
    parser.add_argument('--find-peaks-prominence', default=4, type=int)
    parser.add_argument('--find-peaks-width', default=2, type=float)
    parser.add_argument('--find-peaks-max', default=5, type=int)
    parser.add_argument('-d', '--detector', type=str, required=True)
    parser.add_argument('--maximum-frequency', type=int, default=512)
    parser.add_argument('--minimum-frequency', type=int, default=15)
    parser.add_argument('--buffer-frequency', type=int, default=5)
    parser.add_argument('--spline-min', default=-57, type=int)
    parser.add_argument('--spline-max', default=-32, type=int)
    parser.add_argument('--gamma-min', default=-6, type=float)
    parser.add_argument('--gamma-max', default=0, type=float)
    parser.add_argument('--lorentzian-min', default=-50, type=int)
    parser.add_argument('--lorentzian-max', default=-30, type=int)
    parser.add_argument('--lorentzian-mix', default=0.1, type=float)
    parser.add_argument('--sampler', type=str, default="dynesty")
    parser.add_argument('-c', '--clean', action="store_true")
    parser.add_argument(
        "--sampler-kwargs",
        type=str,
        default="{}",
        help=(
            "Dictionary of sampler-kwargs to pass in, e.g., {nlive: 1000} OR "
            "pass pre-defined set of sampler-kwargs {Default, FastTest}"
        ),
    )
    parser.add_argument('--cpus', type=int, default=1)

    args = parser.parse_args()

    outdir = f'{args.detector}_{args.minimum_frequency}_{args.maximum_frequency}_{args.sampler}'
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)

    # Set up label
    args.nlorentzians_fixed = len(args.fixed_lorentzians)
    label = f'psd_splines{args.nsplines}'
    if args.nlorentzians > 0:
        label += f"_flt{args.nlorentzians}"
    if args.nlorentzians_fixed:
        label += f"_fix{args.nlorentzians_fixed}"
    if args.fixed_lorentzians_file:
        file_line_list = LineList(
            args.fixed_lorentzians_file,
            args.minimum_frequency,
            args.maximum_frequency)
        label += f"_file{len(file_line_list.get_fmin_fmax_list())}"
        fmin_fmax_list = file_line_list.get_fmin_fmax_list()
    else:
        fmin_fmax_list = []
    if args.find_peaks:
        label += f"_fp{args.find_peaks_prominence}-{args.find_peaks_max}-{args.find_peaks_width}"

    # Read in the data
    post_trigger_duration = 2
    analysis_start = args.trigger_time + post_trigger_duration - args.duration

    analysis_data = TimeSeries.fetch_open_data(
        args.detector, analysis_start, analysis_start + args.duration,
        sample_rate=4096, cache=True)

    # Generate the IFO
    ifo = bilby.gw.detector.get_empty_interferometer(args.detector)
    ifo.set_strain_data_from_gwpy_timeseries(analysis_data)
    ifo.minimum_frequency = args.minimum_frequency - args.buffer_frequency
    ifo.maximum_frequency = args.maximum_frequency + args.buffer_frequency

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
        priors[key] = Uniform(args.spline_min, args.spline_max, key, latex_label=latex)

    # Set up floating lorentzian priors
    floating_lorentzian_keys = dict(frequency=[], amplitude=[], gamma=[])
    for ii in range(args.nlorentzians):
        key = f"{ifo.name}_lorentzian_frequency_{ii}"
        latex = f"LF{ii}"
        if ii > 0:
            priors[key] = MinimumPrior(
                order=args.nlorentzians - ii,
                duration=args.duration,
                minimum=args.minimum_frequency,
                maximum=args.maximum_frequency,
                name=key,
                latex_label=latex
            )
            # Hack to fix the class naming
            priors[key].__class__.__name__ = "MinimumPrior"
        else:
            priors[key] = bilby.core.prior.Beta(
                minimum=args.minimum_frequency,
                maximum=args.maximum_frequency,
                alpha=1,
                beta=args.nlorentzians,
                name=key,
                latex_label=latex
            )
        floating_lorentzian_keys["frequency"].append(key)

        key = f"{ifo.name}_lorentzian_amplitude_{ii}"
        priors[key] = get_amplitude_prior(args, key, ii)
        floating_lorentzian_keys["amplitude"].append(key)

        key = f"{ifo.name}_lorentzian_gamma_{ii}"
        priors[key] = get_gamma_prior(args, key, ii)
        floating_lorentzian_keys["gamma"].append(key)

    # Set up fixed lorentzian priors
    lines = [
        line for line in args.fixed_lorentzians
        if ifo.minimum_frequency < line < ifo.maximum_frequency]
    line_width = args.fixed_lorentzians_uncertainty / 2
    fixed_lorentzian_keys = dict(frequency=[], amplitude=[], gamma=[])
    ii = args.nlorentzians
    for line in lines:
        key = f"{ifo.name}_lorentzian_frequency_{ii}"
        latex = f"fixed_LF{ii}"
        priors[key] = Uniform(
            minimum=line - line_width, maximum=line + line_width,
            name=key, latex_label=latex
        )
        fixed_lorentzian_keys["frequency"].append(key)

        key = f"{ifo.name}_lorentzian_amplitude_{ii}"
        priors[key] = get_amplitude_prior(args, key, ii)
        fixed_lorentzian_keys["amplitude"].append(key)

        key = f"{ifo.name}_lorentzian_gamma_{ii}"
        priors[key] = get_gamma_prior(args, key, ii)
        fixed_lorentzian_keys["gamma"].append(key)
        ii += 1

    ii = args.nlorentzians + args.nlorentzians_fixed
    fixed_lorentzian_file_keys = dict(frequency=[], amplitude=[], gamma=[])
    for fmin, fmax in fmin_fmax_list:
        key = f"{ifo.name}_lorentzian_frequency_{ii}"
        latex = f"file_LF{ii}"
        priors[key] = Uniform(
            minimum=fmin, maximum=fmax,
            name=key, latex_label=latex
        )
        fixed_lorentzian_file_keys["frequency"].append(key)

        key = f"{ifo.name}_lorentzian_amplitude_{ii}"
        priors[key] = get_amplitude_prior(args, key, ii)
        fixed_lorentzian_file_keys["amplitude"].append(key)

        key = f"{ifo.name}_lorentzian_gamma_{ii}"
        priors[key] = get_gamma_prior(args, key, ii)
        fixed_lorentzian_file_keys["gamma"].append(key)

        ii += 1

    # Set up peak-found lorentzian priors
    data = np.log(np.abs(ifo.frequency_domain_strain[ifo.frequency_mask]))
    peaks_idxs, properties = find_peaks(
        data, prominence=args.find_peaks_prominence
    )
    if len(peaks_idxs) > args.find_peaks_max:
        peaks_idxs = peaks_idxs[np.argsort(properties['prominences'])]
        peaks_idxs = peaks_idxs[::-1]
        peaks_idxs = peaks_idxs[:args.find_peaks_max]

    find_peaks_lorentzian_keys = dict(frequency=[], amplitude=[], gamma=[])
    for peak in ifo.frequency_array[ifo.frequency_mask][peaks_idxs]:
        key = f"{ifo.name}_lorentzian_frequency_{ii}"
        latex = f"find-peak_LF{ii}"
        priors[key] = Uniform(
            minimum=peak - args.find_peaks_width / 2,
            maximum=peak + args.find_peaks_width / 2,
            name=key, latex_label=latex
        )
        find_peaks_lorentzian_keys["frequency"].append(key)

        key = f"{ifo.name}_lorentzian_amplitude_{ii}"
        priors[key] = get_amplitude_prior(args, key, ii)
        find_peaks_lorentzian_keys["amplitude"].append(key)

        key = f"{ifo.name}_lorentzian_gamma_{ii}"
        priors[key] = get_gamma_prior(args, key, ii)
        find_peaks_lorentzian_keys["gamma"].append(key)

        ii += 1

    farray = np.linspace(ifo.minimum_frequency, ifo.maximum_frequency, 101)
    psd = SplineLorentzianPSD(
        f'{ifo.name}', farray, parameters=priors.sample())

    ifo.power_spectral_density = psd
    likelihood = PSDLikelihood(ifo=ifo)

    plot_data_and_psd_prior(ifo, priors, outdir, label)

    if args.sampler == 'dynesty':
        sampler_kwargs = dict(
            dlogz=.5, queue_size=args.cpus, n_effective=1000, walks=50, nact=2)
    elif args.sampler == 'ptemcee':
        sampler_kwargs = dict(nsamples=1000, ntemps=5, nwalkers=100,
                              threads=args.cpus, pos0='prior')
    else:
        sampler_kwargs = dict()

    sampler_kwargs.update(convert_string_to_dict(args.sampler_kwargs))

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, outdir=outdir, label=label,
        sampler=args.sampler, clean=args.clean, **sampler_kwargs)

    if args.nlorentzians > 0:
        for name, keys in floating_lorentzian_keys.items():
            result.plot_corner(
                keys,
                filename=f"{outdir}/{label}_floating_lorentzian_{name}_corner")

    if args.nlorentzians_fixed > 0:
        for name, keys in fixed_lorentzian_keys.items():
            result.plot_corner(
                keys,
                filename=f"{outdir}/{label}_fixed_lorentzian_{name}_corner")

    if args.fixed_lorentzians_file:
        for name, keys in fixed_lorentzian_file_keys.items():
            result.plot_corner(
                keys,
                filename=f"{outdir}/{label}_file_lorentzian_{name}_corner")

    if args.find_peaks:
        for name, keys in find_peaks_lorentzian_keys.items():
            result.plot_corner(
                keys,
                filename=f"{outdir}/{label}_file_lorentzian_{name}_corner")

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
