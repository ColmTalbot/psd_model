from scipy.interpolate import CubicSpline
import numpy as np

from bilby.gw.detector import PowerSpectralDensity


def cauchy(f, f0, gamma):
    return 1 / np.pi / gamma / (1 + ((f - f0) / gamma)**2)


class SplineLorentzianPSD(PowerSpectralDensity):

    def __init__(self, name, frequency_array, parameters=None):
        self._spline_cache = dict()
        if parameters is None:
            parameters = dict()
        self.name = name
        self.parameters = parameters
        self.frequency_array = frequency_array
        PowerSpectralDensity.__init__(self, frequency_array=frequency_array)
        self.psd_array = self._total_psd(frequency_array)
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
            return self._total_psd(frequency_array)

    def get_amplitude_spectral_density_array(self, frequency_array):
        if self.parameters == self._cache['parameters']:
            return super(SplineLorentzianPSD, self).get_amplitude_spectral_density_array(
                frequency_array=frequency_array)
        else:
            return self._total_psd(frequency_array)

    @property
    def power_spectral_density_interpolated(self):
        return lambda frequency_array: self._total_psd(frequency_array)

    def _total_psd(self, frequency_array):
        return 2 * (
            self.spline(frequency_array=frequency_array)
            + self.lorentzian(frequency_array=frequency_array)
        )

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
    def _lorentzian_qualities_keys(self):
        try:
            self.__lorentzians_qualities_keys = [
                '{}_lorentzian_quality_{}'.format(self.name, ii)
                for ii in range(self.n_lorentzians)]
            return self.__lorentzians_qualities_keys
        except AttributeError:
            return None

    @property
    def lorentzian_qualities(self):
        return np.array([self.parameters[key] for key in self._lorentzian_qualities_keys])

    @property
    def spline_interpolant(self):
        if not (
            self._spline_cache.get("frequencies", None) == self.spline_frequencies
            and self._spline_cache.get("amplitudes", None) == self.spline_amplitudes
            and isinstance(self._spline_cache.get("interpolant", None), CubicSpline)
        ):
            self._spline_cache["interpolant"] = CubicSpline(
                self.spline_frequencies, self.spline_amplitudes
            )
            self._spline_cache["frequencies"] = self.spline_frequencies
            self._spline_cache["amplitudes"] = self.spline_amplitudes
        return self._spline_cache["interpolant"]

    def spline(self, frequency_array):
        if self.n_points > 0:
            exponent = self.spline_interpolant(frequency_array)
            return np.power(10, exponent)
        elif self.n_points == 0:
            return np.zeros_like(frequency_array)

    def lorentzian(self, frequency_array):
        if self.n_lorentzians == 0:
            return 0 * frequency_array
        else:
            aas = self.lorentzian_amplitudes
            qqs = self.lorentzian_qualities
            ffs = self.lorentzian_frequencies
            lorentzian = np.sum(
                self._single_lorentzian(
                    frequency_array[:, np.newaxis],
                    10.**aas * self.spline(ffs),
                    10.**qqs,
                    ffs
                ),
                axis=-1)
            return lorentzian

    @staticmethod
    def _single_lorentzian(frequency_array, amplitude, quality, location):
        return cauchy(f=frequency_array, f0=location, gamma=quality) * amplitude
