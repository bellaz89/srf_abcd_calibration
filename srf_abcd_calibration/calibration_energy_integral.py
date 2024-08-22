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
from .calibration_impl import calibrate_energy_integral
from .find_hbw_decay import find_hbw_decay

# Calibrate using the energy constrained method
def calibration_energy_integral(fs, probe_cmplx, vforw_cmplx, vrefl_cmplx,
                                decay_time, transition_guard=100,
                                start_time=0.0, stop_time=None):

    # Time step
    dt = 1.0/fs
    # Time around the transitions to ignore
    time_guard = transition_guard * dt
    trace_length = probe_cmplx.shape[0]
    time_trace = np.linspace(0, trace_length*dt, trace_length)

    if stop_time == None:
        stop_time = dt * trace_length

    # Indices arrays for filling flattop and decay
    start_time += time_guard
    decay_time += time_guard
    stop_time -= time_guard

    decay_idxs = ((time_trace >= decay_time + time_guard) *
                  (time_trace < stop_time - time_guard)).astype(bool)

    pulse_idxs = ((time_trace >= time_guard + start_time) *
                  (time_trace < stop_time - time_guard)).astype(bool)

    time_trace_decay = time_trace[decay_idxs]
    probe_cmplx_decay = probe_cmplx[decay_idxs]

    time_trace_pulse = time_trace[pulse_idxs]
    probe_cmplx_pulse = probe_cmplx[pulse_idxs]
    vforw_cmplx_pulse = vforw_cmplx[pulse_idxs]
    vrefl_cmplx_pulse = vrefl_cmplx[pulse_idxs]

    #Half bandwidth (in angular frequency) computed on the probe decay
    hbw_decay = find_hbw_decay(time_trace_decay, probe_cmplx_decay)
    abcd = calibrate_energy_integral(fs,
                                     hbw_decay,
                                     decay_time,
                                     probe_cmplx_pulse,
                                     vforw_cmplx_pulse,
                                     vrefl_cmplx_pulse)

    return abcd, hbw_decay
