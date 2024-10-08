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
from abc import ABC, abstractmethod
from .calibration_energy_constr import calibrate_energy_constr
from .calibration_impl import calibrate_diagonal
from .find_klfd import find_klfd
from .find_bwdet import find_bwdet

# Amplitude-phase to complex
def AP2C(amplitude, phase):
    return amplitude * np.exp(1.0j * phase * np.pi / 180)

# Complex to amplitude-phase
def C2AP(cmplx):
    return np.abs(cmplx), np.angle(cmplx, deg=True)

# Angular frequency to frequency
def ANG2HZ(ang):
    return ang / (2.0 * np.pi)

class Station(ABC):
    def __init__(self, name, conf):
        self.name = conf.get("name", name)
        self.conf = conf
        self.transition_guard = conf.get("transition_guard", 100)

        # offset from the start of the filling in (s). Used to remove noisy data
        self.start_time = conf.get("start_time", 0)

        # maximum trace length to calibrate in s
        self.stop_time = conf.get("stop_time", None)

        # max adc value scaling
        self.max_adc_scaling = conf.get("max_adc_scaling", 2.0)

        # min adc value scaling
        self.min_adc_scaling = conf.get("min_adc_scaling", 0.1)

        # max abcd scaling
        self.max_abcd_scaling = conf.get("max_abcd_scaling", 2.0)

        # plot scale for detuning
        self.plot_det_scale = conf.get("plot_det_scale", [-750, 750])

        # plot scale for bandwidth
        self.plot_hbw_scale = conf.get("plot_hbw_scale", [-250, 1250])

    # Check if the type can be loaded by checking e.g. if the correct communication
    # library is intalled
    @staticmethod
    def loadable():
        return False

    # return the cavity resonance frequency(Hz), the sampling frequency(Hz),
    # the probe, vforw, and vrefl in complex notation
    # the flattop start time, the decay time
    @abstractmethod
    def get_rf_traces_params(self):
        pass

    # get the abcd scaling from hardware, complex
    @abstractmethod
    def get_abcd_scaling(self):
        pass

    # set the abcd scaling to hardware, complex
    @abstractmethod
    def set_abcd_scaling(self, a, b, c, d):
        pass

    # get the half bandwidth to hardware
    @abstractmethod
    def get_hbw_decay(self):
        pass

    # set the half bandwidth to hardware
    @abstractmethod
    def set_hbw_decay(seĺf, hbw):
        pass

    # get the forward and reflected (XY) scaling from hardware, complex
    @abstractmethod
    def get_xy_scaling(self):
        pass

    # set the forward and reflected (XY) scaling to hardware, complex
    @abstractmethod
    def set_xy_scaling(self, x, y):
        pass

    # get the probe scaling from hardware, complex
    @abstractmethod
    def get_probe_amplitude_scaling(self):
        pass

    # set the probe scaling from hardware, complex
    @abstractmethod
    def set_probe_amplitude_scaling(self, scale):
        pass

    # get the cavity voltage as scalar from hardware (MV)
    # this method can return None if such a device is not available
    @abstractmethod
    def get_cavity_voltage(self):
        pass

    # Calibrate the RF signals using the diagonal method
    def calibrate_xy(self, probe, vforw, vrefl):
        return calibrate_diagonal(probe,
                                  vforw,
                                  vrefl)

    # Calibrate the RF signals using the energy constrained method
    def calibrate_abcd(self, fs, probe_cmplx, vforw_cmplx, vrefl_cmplx,
                       flattop_start, decay_start):

        return calibrate_energy_constr(fs,
                                       probe_cmplx,
                                       vforw_cmplx,
                                       vrefl_cmplx,
                                       flattop_start,
                                       decay_start,
                                       transition_guard=self.transition_guard,
                                       start_time=self.start_time,
                                       stop_time=self.stop_time)

    # Calibrate  the RF signals
    def calibrate(self):
        [f0,
        fs,
        probe_cmplx,
        vforw_cmplx,
        vrefl_cmplx,
        flattop_start,
        decay_start] = self.get_rf_traces_params()

        # Rescale the probe amplitude
        voltage = self.get_cavity_voltage()

        if voltage:
            probe_abs = np.abs(probe_cmplx)
            probe_scaling = voltage / np.max(probe_abs)

            probe_cmplx *= probe_scaling

            probe_scaling *= self.get_probe_amplitude_scaling()
            probe_scaling = np.clip(probe_scaling, self.min_adc_scaling, self.max_adc_scaling)
            self.set_probe_amplitude_scaling(probe_scaling)


        # Perform XY calibration
        diag = self.calibrate_xy(probe_cmplx, vforw_cmplx, vrefl_cmplx)
        vforw_cmplx *= diag[0]
        vrefl_cmplx *= diag[3]

        (x, y)  = self.get_xy_scaling()
        x *= diag[0]
        y *= diag[3]

        (x_amp, x_pha) = C2AP(x)
        (y_amp, y_pha) = C2AP(y)

        [x_amp, y_amp] = np.clip([x_amp, y_amp], self.min_adc_scaling, self.max_adc_scaling)

        x = AP2C(x_amp, x_pha)
        y = AP2C(y_amp, y_pha)

        self.set_xy_scaling(x, y)

        abcd, hbw_decay = self.calibrate_abcd(fs,
                                              probe_cmplx,
                                              vforw_cmplx,
                                              vrefl_cmplx,
                                              flattop_start,
                                              decay_start)

        self.set_hbw_decay(hbw_decay)

        re_abcd = np.real(abcd)
        im_abcd = np.imag(abcd)

        re_abcd = np.clip(re_abcd, -self.max_abcd_scaling, self.max_abcd_scaling)
        im_abcd = np.clip(im_abcd, -self.max_abcd_scaling, self.max_abcd_scaling)
        abcd = re_abcd + 1.0j * im_abcd

        self.set_abcd_scaling(*abcd)

    def get_ui_data(self):
        [f0,
        fs,
        probe_cmplx,
        vforw_cmplx,
        vrefl_cmplx,
        flattop_start,
        decay_start] = self.get_rf_traces_params()

        result = dict()
        rf_traces_params = self.get_rf_traces_params()

        hbw_decay = self.get_hbw_decay()

        result["f0"] = f0
        result["fs"] = fs
        result["xy"] = self.get_xy_scaling()
        result["abcd"] = self.get_abcd_scaling()
        abcd_tot = np.array(result["abcd"])
        abcd_tot[0] *= result["xy"][0]
        abcd_tot[1] *= result["xy"][1]
        abcd_tot[2] *= result["xy"][0]
        abcd_tot[3] *= result["xy"][1]
        result["abcd_tot"] = abcd_tot
        result["hbw_decay"] = ANG2HZ(hbw_decay)
        result["QL"] = f0 / ANG2HZ(2.0 * hbw_decay)

        bwdetXY = find_bwdet(fs, hbw_decay,
                             probe_cmplx, vforw_cmplx, self.transition_guard)

        vforw_cmplx_corr = result["abcd"][0] * vforw_cmplx + result["abcd"][1] * vrefl_cmplx
        vrefl_cmplx_corr = result["abcd"][2] * vforw_cmplx + result["abcd"][3] * vrefl_cmplx

        bwdetABCD =  find_bwdet(fs, hbw_decay,
                                probe_cmplx, vforw_cmplx_corr, self.transition_guard)

        (hbwXY, detXY) = bwdetXY
        (hbwABCD, detABCD) = bwdetABCD

        time_trace = np.linspace(0, len(probe_cmplx)/fs, len(probe_cmplx))

        (det0, klfd, slope) = find_klfd(time_trace, probe_cmplx,
                                        detABCD, self.transition_guard)

        result["klfd"] = ANG2HZ(klfd)
        result["det0"] = ANG2HZ(det0)
        result["slope"] = ANG2HZ(slope)
        result["peak_amp"] = np.max(np.abs(probe_cmplx))

        result["time_trace"] = time_trace

        result["probe_I"] = np.real(probe_cmplx)
        result["probe_Q"] = np.imag(probe_cmplx)

        VprobeXY = vforw_cmplx + vrefl_cmplx
        VprobeABCD = vforw_cmplx_corr + vrefl_cmplx_corr

        result["VprobeXY_I"] = np.real(VprobeXY)
        result["VprobeXY_Q"] = np.imag(VprobeXY)
        result["VprobeABCD_I"] = np.real(VprobeABCD)
        result["VprobeABCD_Q"] = np.imag(VprobeABCD)

        result["probe_amp"] = np.abs(probe_cmplx)
        result["vforwXY_amp"] = np.abs(vforw_cmplx)
        result["vreflXY_amp"] = np.abs(vrefl_cmplx)
        result["vforwABCD_amp"] = np.abs(vforw_cmplx_corr)
        result["vreflABCD_amp"] = np.abs(vrefl_cmplx_corr)

        result["hbwDECAY"] = ANG2HZ(hbw_decay * np.ones(len(hbwABCD)))
        result["hbwXY"] = ANG2HZ(hbwXY)
        result["hbwABCD"] = ANG2HZ(hbwABCD)

        result["detXY"] = ANG2HZ(detXY)
        result["detABCD"] = ANG2HZ(detABCD)
        result["detEST"] = ANG2HZ(det0 +
                                  klfd * np.abs(probe_cmplx) ** 2 +
                                  slope * time_trace)

        result["transition_guard"] = self.transition_guard
        result["start_time"] = self.start_time
        result["stop_time"] = self.stop_time

        return result

