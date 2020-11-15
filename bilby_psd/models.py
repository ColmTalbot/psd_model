from __future__ import division

from scipy.interpolate import CubicSpline
import numpy as np

from bilby.core.prior import Prior
from bilby.gw.detector import PowerSpectralDensity


def cauchy(f, f0, gamma):
    return 1 / np.pi / (1 + ((f - f0) / gamma)**2)


def z_function(f, f0):
    fprime = np.abs(f - f0)
    df = f0 / 50
    z = np.ones_like(fprime)
    idxs = fprime > df
    z[idxs] = np.exp(- (fprime - df) / df)[idxs]
    return z


def Lambda(f, f0, gamma):
    z = z_function(f, f0)
    denom = (f0 * f) ** 2 + ((f0 ** 2 - f ** 2) / gamma) ** 2
    return z * f0 ** 4 / denom


class SplineLorentzianPSD(PowerSpectralDensity):

    def __init__(self, name, frequency_array, parameters=None):
        if parameters is None:
            parameters = dict()
        self.name = name
        self.parameters = parameters
        self.frequency_array = frequency_array
        PowerSpectralDensity.__init__(self, frequency_array=frequency_array)
        self.psd_array = (self.spline(self.frequency_array) + self.lorentzian(self.frequency_array))
        self._cache['parameters'] = dict()

    def _update_cache(self, frequency_array):
        psd_array = self.power_spectral_density_interpolated(frequency_array)
        self._cache['psd_array'] = psd_array
        self._cache['asd_array'] = psd_array**0.5
        self._cache['frequency_array'] = frequency_array
        self._cache['parameters'] = self.parameters.copy()

    def get_power_spectral_density_array(self, frequency_array):

        if self.parameters == self._cache['parameters']:
            return super(SplineLorentzianPSD, self).get_power_spectral_density_array(
                frequency_array=frequency_array)
        else:
            return 2 * (self.spline(frequency_array=frequency_array) +
                        self.lorentzian(frequency_array=frequency_array))

    def get_amplitude_spectral_density_array(self, frequency_array):
        if self.parameters == self._cache['parameters']:
            return super(SplineLorentzianPSD, self).get_amplitude_spectral_density_array(
                frequency_array=frequency_array)
        else:
            return np.sqrt(self.get_power_spectral_density_array(frequency_array))

    @property
    def power_spectral_density_interpolated(self):
        return lambda frequency_array: 2 * (
            self.spline(frequency_array=frequency_array) +
            self.lorentzian(frequency_array=frequency_array))

    @property
    def n_points(self):
        try:
            return self.parameters['{}_n_spline'.format(self.name)]
        except KeyError:
            return len(
                [key for key in self.parameters
                 if '{}_spline_frequency'.format(self.name) in key])

    @property
    def _spline_frequencies_keys(self):
        try:
            return self.__spline_frequencies_keys
        except AttributeError:
            self.__spline_frequencies_keys = [
                '{}_spline_frequency_{}'.format(self.name, ii)
                for ii in range(self.n_points)]
            return self.__spline_frequencies_keys

    @property
    def spline_frequencies(self):
        return [self.parameters[key] for key in self._spline_frequencies_keys]

    @property
    def _spline_amplitude_keys(self):
        try:
            return self.__spline_amplitude_keys
        except AttributeError:
            self.__spline_amplitude_keys = [
                '{}_spline_amplitude_{}'.format(self.name, ii)
                for ii in range(self.n_points)]
            return self.__spline_amplitude_keys

    @property
    def spline_amplitudes(self):
        return [self.parameters[key] for key in self._spline_amplitude_keys]

    @property
    def n_lorentzians(self):
        try:
            return int(self.parameters['{}_n_lorentzians'.format(self.name)])
        except KeyError:
            return len(
                [key for key in self.parameters
                 if '{}_lorentzian_frequency'.format(self.name) in key])

    @property
    def _lorentzian_frequencies_keys(self):
        try:
            self.__lorentzian_frequencies_keys = [
                '{}_lorentzian_frequency_{}'.format(self.name, ii)
                for ii in range(self.n_lorentzians)]
            return self.__lorentzian_frequencies_keys
        except AttributeError:
            return None

    @property
    def lorentzian_frequencies(self):
        return np.array([self.parameters[key] for key in self._lorentzian_frequencies_keys])

    @property
    def _lorentzian_amplitudes_keys(self):
        try:
            self.__lorentzian_amplitudes_keys = [
                '{}_lorentzian_amplitude_{}'.format(self.name, ii)
                for ii in range(self.n_lorentzians)]
            return self.__lorentzian_amplitudes_keys
        except AttributeError:
            return None

    @property
    def lorentzian_amplitudes(self):
        return np.array([self.parameters[key] for key in self._lorentzian_amplitudes_keys])

    @property
    def _lorentzian_gamma_keys(self):
        try:
            self.__lorentzians_gamma_keys = [
                '{}_lorentzian_gamma_{}'.format(self.name, ii)
                for ii in range(self.n_lorentzians)]
            return self.__lorentzians_gamma_keys
        except AttributeError:
            return None

    @property
    def lorentzian_gamma(self):
        return np.array([self.parameters[key] for key in self._lorentzian_gamma_keys])

    def spline(self, frequency_array):
        if self.n_points > 0:
            interp = CubicSpline(self.spline_frequencies,
                                 self.spline_amplitudes)
            exponent = interp(frequency_array)
            return np.power(10, exponent)
        elif self.n_points == 0:
            return np.zeros_like(frequency_array)

    def lorentzian(self, frequency_array):
        if self.n_lorentzians == 0:
            return 0 * frequency_array
        else:
            aas = self.lorentzian_amplitudes
            qqs = self.lorentzian_gamma
            ffs = self.lorentzian_frequencies
            lorentzian = np.sum(
                self._single_lorentzian(
                    frequency_array[:, np.newaxis], 10.**aas, 10 ** qqs, ffs),
                axis=-1)
            return lorentzian

    @staticmethod
    def _single_lorentzian(frequency_array, amplitude, gamma, location):
        return Lambda(f=frequency_array, f0=location, gamma=gamma) * amplitude


class Discrete(Prior):

    def __init__(self, minimum, maximum, step_size, name=None,
                 latex_label=None, boundary=None):
        super(Discrete, self).__init__(
            name=name, latex_label=latex_label, boundary=boundary)
        self.minimum = minimum
        self.maximum = maximum
        self.step_size = step_size
        if (maximum - minimum + 1) % step_size != 0:
            raise ValueError(
                'maximum - minimum must be an integer multiple of step size')

    @property
    def n_bins(self):
        return (self.maximum - self.minimum + 1) / self.step_size

    def prob(self, val):
        prob = 1 / self.n_bins
        return prob

    def rescale(self, val):
        val = np.atleast_1d(val)
        val *= self.step_size * self.n_bins
        val += self.minimum
        return val.astype(int)
