#!/usr/bin/env python
"""

"""
from __future__ import division, print_function

import numpy as np
import bilby
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

from models import SplineLorentzianPSD
from likelihood import PSDLikelihood

GW150914 = 1126259462
trigger_time = GW150914 - 20
ifo = bilby.gw.detector.get_interferometer_with_open_data(
    'H1', trigger_time=trigger_time)

x = ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask]
y = ifo.strain_data.frequency_domain_strain[ifo.strain_data.frequency_mask]

idxs = x < 2**10
x = x[idxs]
y = y[idxs]

logx = np.log10(x)
logy = np.log10(y)

logysmooth = savgol_filter(logy, window_length=15, polyorder=9)
peaks, _ = find_peaks(logysmooth, prominence=1.1,)

plt.plot(logx, logy)
plt.plot(logx, logysmooth, lw=0.5)
plt.plot(logx[peaks], logy[peaks], 'rx')
plt.savefig('peaks')

with open('peak_freqs.txt', 'w+') as file:
    for f in x[peaks]:
        print(f, file=file)
