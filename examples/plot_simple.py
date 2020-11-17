import matplotlib.pyplot as plt
import numpy as np
from bilby_psd.models import SplineLorentzianPSD
from scipy.interpolate import CubicSpline

nsplines = 20
minimum_frequency = 20
maximum_frequency = 1024
farray = np.linspace(minimum_frequency, maximum_frequency, 100000)

fixed_spline_points = np.logspace(
    np.log10(minimum_frequency),
    np.log10(maximum_frequency),
    nsplines)

fixed_spline_amplitudes = np.log10(1e-26 * (fixed_spline_points - 500)**2)

parameters =dict()
for i, (f, a) in enumerate(zip(fixed_spline_points, fixed_spline_amplitudes)):
    parameters[f"H1_spline_frequency_{i}"] = f
    parameters[f"H1_spline_amplitude_{i}"] = a

parameters["H1_lorentzian_frequency_0"] = 100
parameters["H1_lorentzian_amplitude_0"] = -20
parameters["H1_lorentzian_gamma_0"] = -3.

psd = SplineLorentzianPSD('H1', farray, parameters=parameters)

fig, ax = plt.subplots()
ax.loglog(farray, psd.get_amplitude_spectral_density_array(farray))
ax.loglog(fixed_spline_points, np.power(10., fixed_spline_amplitudes), 'x', color='C3')
fig.savefig("plot", dpi=500)
