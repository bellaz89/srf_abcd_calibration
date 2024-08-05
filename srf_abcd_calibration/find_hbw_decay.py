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
from scipy.optimize import curve_fit

#Find the half bandwidth using tje field decay
def find_hbw_decay(time_decay, probe_cmplx_decay):
    probe_amp = np.abs(probe_cmplx_decay)


    def decay(t, a, b, c):
        return a * np.exp(-(t-time_decay[0]) * b) + c

    popt, pcov = curve_fit(decay,
                           time_decay, probe_amp,
                           p0=[probe_amp[0], np.pi * 130, 0])

    return popt[1]
