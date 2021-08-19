import numpy as np
from scipy.special import logsumexp

from bilby.core.likelihood import Likelihood
from bilby.gw.likelihood import GravitationalWaveTransient
from pygwb.spectral import coarse_grain


class PSDLikelihood(Likelihood):

    def __init__(self, ifo, coarsen=1):
        super(PSDLikelihood, self).__init__(
            parameters=ifo.power_spectral_density.parameters
        )
        self.ifo = ifo
        self.coarsen = coarsen
        self.weight = 2 / ifo.strain_data.duration
        self.mask = coarse_grain(
            self.ifo.frequency_mask.astype(float),
            coarsening_factor=coarsen
        )
        self.keep = (self.mask == 1).astype(bool)
        self.frequency_array = coarse_grain(
            self.ifo.frequency_array, coarsening_factor=coarsen
        )[self.keep]
        self.data = coarse_grain(
            abs(ifo.frequency_domain_strain)**2,
            coarsening_factor=coarsen
        )[self.keep]
        self.mask = self.mask[self.keep]
        naive_psd = coarse_grain(
            abs(ifo.frequency_domain_strain) ** 2,
            coarsening_factor=coarsen
        )[self.keep]
        self._nll = - sum(self.mask) / 2 - np.sum(np.log(
            2 * np.pi * naive_psd)) * self.coarsen

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood()

    def log_likelihood(self):
        psd = self.psd / self.weight
        ln_l = - np.sum(self.data / psd + np.log(2 * np.pi * psd))
        # return ln_l
        return ln_l * self.coarsen

    def noise_log_likelihood(self):
        return self._nll

    @property
    def psd(self):
        return self.ifo.power_spectral_density.get_power_spectral_density_array(
            self.frequency_array
        )

    @property
    def asd(self):
        return self.psd**0.5


class PSDGravitationalWaveTransient(GravitationalWaveTransient):

    def __init__(
        self, ifos, wfg, priors=None, distance_marginalization=None,
        phase_marginalization=None, time_marginalization=None,
        reference_frame="sky", time_reference="geocent"
    ):
        GravitationalWaveTransient.__init__(
            self, interferometers=ifos, waveform_generator=wfg, priors=priors,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            reference_frame=reference_frame,
            time_reference=time_reference
        )
        self._nll = np.nan
        for ifo in self.interferometers:
            ifo.power_spectral_density.parameters = self.parameters

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood()

    def log_likelihood(self):
        log_l = GravitationalWaveTransient.log_likelihood_ratio(self)
        for ifo in self.interferometers:
            log_l -= np.sum(np.log(2 * np.pi * ifo.power_spectral_density_array[
                ifo.frequency_mask]))
            log_l -= 2. / self.waveform_generator.duration * np.sum(
                abs(ifo.frequency_domain_strain[ifo.frequency_mask]) ** 2 /
                ifo.power_spectral_density_array[ifo.frequency_mask])
        return log_l

    def noise_log_likelihood(self):
        if np.isnan(self._nll):
            self._nll = 0
            for ifo in self.interferometers:
                self._nll += - sum(ifo.frequency_mask) / 2 - np.sum(
                    np.log(
                        2 * np.pi * abs(
                            ifo.frequency_domain_strain[ifo.frequency_mask]
                        )**2
                    )
                )
        return self._nll


class ManyPSDGravitationalWaveTransient(GravitationalWaveTransient):

    def __init__(
            self, ifos, wfg, psds, priors=None, distance_marginalization=None,
            phase_marginalization=None, time_marginalization=None,
            reference_frame="sky", time_reference="geocent",
    ):
        super(ManyPSDGravitationalWaveTransient, self).__init__(
            interferometers=ifos, waveform_generator=wfg, priors=priors,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization,
            reference_frame=reference_frame,
            time_reference=time_reference
        )
        self._nll = None
        self.psds = dict()
        self.ln_psd_weights = 0
        for ifo in self.interferometers:
            self.psds[ifo.name] = psds[ifo.name][ifo.frequency_mask]
            self.ln_psd_weights = self.ln_psd_weights - np.sum(np.log(
                self.psds[ifo.name]
            ), axis=0)
            mask = ifo.frequency_mask
            self.ln_psd_weights -= np.dot(
                abs(ifo.frequency_domain_strain.conjugate()[mask]) ** 2,
                1 / self.psds[ifo.name]
            ) / 2
        # self.ln_psd_weights -= max(self.ln_psd_weights)
        self.ln_psd_weights -= np.log(len(self.ln_psd_weights))

    def calculate_snrs(self, waveform_polarizations, interferometer):
        """
        Compute the snrs

        Parameters
        ----------
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object

        """
        mask = interferometer.frequency_mask
        signal = interferometer.get_detector_response(
            waveform_polarizations, self.parameters
        )[mask]
        d_inner_h = np.dot(
            interferometer.frequency_domain_strain.conjugate()[mask] * signal,
            1 / self.psds[interferometer.name]
        )
        optimal_snr_squared = np.dot(
            abs(signal) ** 2,
            1 / self.psds[interferometer.name]
        )
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        if self.time_marginalization:
            d_inner_h_squared_tc_array =\
                4 / self.waveform_generator.duration * np.fft.fft(
                    signal[0:-1] *
                    interferometer.frequency_domain_strain.conjugate()[0:-1] /
                    interferometer.power_spectral_density_array[0:-1])
        else:
            d_inner_h_squared_tc_array = None

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=d_inner_h_squared_tc_array)

    def log_likelihood_ratio(self):
        return np.nan

    def log_likelihood(self):
        waveform_polarizations =\
            self.waveform_generator.frequency_domain_strain(self.parameters)

        self.parameters.update(self.get_sky_frame_parameters())

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        log_l = self.ln_psd_weights.copy()
        d_inner_h = 0.
        optimal_snr_squared = 0.
        complex_matched_filter_snr = 0.

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer)

            d_inner_h += per_detector_snr.d_inner_h
            optimal_snr_squared += np.real(per_detector_snr.optimal_snr_squared)
            complex_matched_filter_snr += per_detector_snr.complex_matched_filter_snr

        if self.distance_marginalization:
            log_l += self.distance_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=optimal_snr_squared
            )
        elif self.phase_marginalization:
            log_l += self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=optimal_snr_squared
            )
        else:
            log_l += np.real(d_inner_h) - optimal_snr_squared / 2
        self._log_ls = log_l

        log_l = logsumexp(np.real(log_l))
        return log_l

    def noise_log_likelihood(self):
        return logsumexp(self.ln_psd_weights)

    def generate_posterior_sample_from_marginalized_likelihood(self):
        return self.parameters.copy()
