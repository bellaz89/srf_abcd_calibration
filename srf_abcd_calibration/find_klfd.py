# SRF ABCD calibration program for superconducting accelerating cavity's
# LLRF system
# Copyright (C) 2024 Andrea Bellandi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def find_klfd(time_trace, probe_cmplx, detuning):
    probe_amp_sq = np.abs(probe_cmplx)**2

    def klfd_square(t, det0, klfd, det_slope):
        return det0 + klfd * np.interp(t, time_trace, probe_amp_sq) + det_slope * t

    popt, pcov = curve_fit(klfd_square, time_trace, detuning, p0 = [0, -1, 0])
    return (popt[0], popt[1], popt[2])


