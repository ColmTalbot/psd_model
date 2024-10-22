from __future__ import division

from scipy.interpolate import CubicSpline
import numpy as np

from bilby.core.prior import Prior
from bilby.gw.detector import PowerSpectralDensity


def cauchy(f, f0, gamma):
    return 1 / np.pi / (1 + ((f - f0) / gamma)**2)


class SplineLorentzianPSD(PowerSpectralDensity):

    def __init__(self, name, frequency_array, parameters=None):
        if parameters is None:
            parameters = dict()
        self.name = name
        self.parameters = parameters
        self.frequency_array = frequency_array
        PowerSpectralDensity.__init__(self, frequency_array=frequency_array)
        self.psd_array = (
                self.spline(self.frequency_array) +
                self.lorentzian(self.frequency_array))

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
            # return self.__lorentzian_frequencies_keys
        # except AttributeError:
            self.__lorentzian_frequencies_keys = [
                '{}_lorentzian_frequency_{}'.format(self.name, ii)
                for ii in range(self.n_lorentzians)]
            return self.__lorentzian_frequencies_keys
        except AttributeError:
            return None

    @property
    def lorentzian_frequencies(self):
        return [self.parameters[key] for key in self._lorentzian_frequencies_keys]

    @property
    def _lorentzian_amplitudes_keys(self):
        try:
            # return self.__lorentzian_amplitudes_keys
        # except AttributeError:
            self.__lorentzian_amplitudes_keys = [
                '{}_lorentzian_amplitude_{}'.format(self.name, ii)
                for ii in range(self.n_lorentzians)]
            return self.__lorentzian_amplitudes_keys
        except AttributeError:
            return None

    @property
    def lorentzian_amplitudes(self):
        return [self.parameters[key] for key in self._lorentzian_amplitudes_keys]

    @property
    def _lorentzian_qualities_keys(self):
        try:
            # return self.__lorentzians_qualities_keys
        # except AttributeError:
            self.__lorentzians_qualities_keys = [
                '{}_lorentzian_quality_{}'.format(self.name, ii)
                for ii in range(self.n_lorentzians)]
            return self.__lorentzians_qualities_keys
        except AttributeError:
            return None

    @property
    def lorentzian_qualities(self):
        return [self.parameters[key] for key in self._lorentzian_qualities_keys]

    def spline(self, frequency_array):
        if self.n_points > 0:
            return 10 ** CubicSpline(
                self.spline_frequencies, self.spline_amplitudes)(frequency_array)
        elif self.n_points is 0:
            return np.zeros_like(frequency_array)

    def lorentzian(self, frequency_array):
        lorentzian = 0 * frequency_array
        # print(self.n_lorentzians)
        if self.n_lorentzians == 0:
            return lorentzian
        else:
            aas = self.lorentzian_amplitudes
            qqs = self.lorentzian_qualities
            ffs = self.lorentzian_frequencies
            # print(qqs, aas, ffs)
            for ii in range(self.n_lorentzians):
                lorentzian += self._single_lorentzian(
                    frequency_array, 10 ** aas[ii], 10 ** qqs[ii], ffs[ii])
            return lorentzian

    @staticmethod
    def _single_lorentzian(frequency_array, amplitude, quality, location):
        return cauchy(f=frequency_array, f0=location, gamma=quality) * amplitude


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
