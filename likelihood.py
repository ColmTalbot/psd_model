import numpy as np

from bilby.core.likelihood import Likelihood
from bilby.gw.likelihood import GravitationalWaveTransient


class PSDLikelihood(Likelihood):

    def __init__(self, ifo):
        Likelihood.__init__(
            self, parameters=ifo.power_spectral_density.parameters)
        self.ifo = ifo
        self.data = abs(ifo.frequency_domain_strain)**2
        self.data = self.data[self.ifo.frequency_mask]
        self.weight = 2 / ifo.strain_data.duration

    def log_likelihood(self):
        return - np.sum((self.weight * self.data / self.psd +
                         np.log(2 * np.pi * self.psd))) / 2

    @property
    def psd(self):
        return self.ifo.power_spectral_density.get_power_spectral_density_array(
            self.ifo.frequency_array[self.ifo.frequency_mask])

    @property
    def asd(self):
        return self.psd**0.5


class PSDGravitationalWaveTransient(GravitationalWaveTransient):

    def __init__(self, ifos, wfg, priors=None, distance_marginalization=None,
                 phase_marginalization=None, time_marginalization=None):
        GravitationalWaveTransient.__init__(
            self, interferometers=ifos, waveform_generator=wfg, priors=priors,
            distance_marginalization=distance_marginalization,
            phase_marginalization=phase_marginalization,
            time_marginalization=time_marginalization)

    def log_likelihood(self):
        log_l = GravitationalWaveTransient.log_likelihood_ratio(self)
        for ifo in self.interferometers:
            log_l -= np.sum(np.log(2 * np.pi * ifo.power_spectral_density_array[
                ifo.frequency_mask])) / 2.0
            log_l -= 2. / self.waveform_generator.duration * np.sum(
                abs(ifo.frequency_domain_strain[ifo.frequency_mask]) ** 2 /
                ifo.power_spectral_density_array[ifo.frequency_mask])
        return log_l
