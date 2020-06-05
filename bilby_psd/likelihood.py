import numpy as np

from bilby.core.likelihood import Likelihood
from bilby.gw.likelihood import GravitationalWaveTransient


class PSDLikelihood(Likelihood):

    def __init__(self, ifo):
        super(PSDLikelihood, self).__init__(
            parameters=ifo.power_spectral_density.parameters
        )
        self.ifo = ifo
        self.data = abs(ifo.frequency_domain_strain[self.ifo.frequency_mask])**2
        self._nll = - sum(self.ifo.frequency_mask) / 2 - np.sum(np.log(
            2 * np.pi * abs(
                ifo.frequency_domain_strain[self.ifo.frequency_mask])**2)
        )
        self.weight = 2 / ifo.strain_data.duration

    def log_likelihood_ratio(self):
        return self.log_likelihood() - self.noise_log_likelihood()

    def log_likelihood(self):
        fdiff = np.diff(self.ifo.power_spectral_density.lorentzian_frequencies)
        if np.any(fdiff < 0):
            return -np.nan_to_num(np.inf)

        psd = self.psd / self.weight
        return - np.sum(self.data / psd + np.log(2 * np.pi * psd))

    def noise_log_likelihood(self):
        return self._nll

    @property
    def psd(self):
        return self.ifo.power_spectral_density.get_power_spectral_density_array(
            self.ifo.frequency_array[self.ifo.frequency_mask])

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
