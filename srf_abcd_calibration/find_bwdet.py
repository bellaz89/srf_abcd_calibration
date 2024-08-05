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
# along with this program.  If not, see <https://www.gnu.org/licens/>.

import numpy as np
from scipy.signal import savgol_filter
import warnings

ORDER_SAVGOL = 2

# Find bandwidth and detuning using the inverse method
def find_bwdet(fs, hbw_decay, probe_cmplx, vforw_cmplx, transition_guard):

    probe_deriv = np.gradient(probe_cmplx) * fs
    A2 = np.abs(probe_cmplx) ** 2
    rt = 2.0 * hbw_decay * vforw_cmplx - probe_deriv

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bwdet = probe_cmplx * np.conj(rt) / A2
        return (savgol_filter(np.real(bwdet), transition_guard, ORDER_SAVGOL),
                -savgol_filter(np.imag(bwdet), transition_guard, ORDER_SAVGOL))


