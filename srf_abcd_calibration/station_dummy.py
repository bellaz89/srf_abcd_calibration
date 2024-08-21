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

from .station import Station
import numpy as np
from numpy.random import normal
from numpy.linalg import inv
from scipy.integrate import ode

class DummyStation(Station):
    @staticmethod
    def loadable():
        return True

    def __init__(self, name, conf):
        super(DummyStation, self).__init__(name, conf)
        self.f0                  = self.conf.get("f0", 1.3e9)
        self.fs                  = self.conf.get("fs", 10e6)
        self.flattop_start       = self.conf.get("flattop_start", 1e-3)
        self.decay_start         = self.conf.get("decay_start", 2e-3)
        self.decay_stop          = self.conf.get("decay_stop", 3e-3)
        self.vforw_flattop       = self.conf.get("vforw_flattop", 7.5)
        self.hbw                 = self.conf.get("hbw", 65.0)
        self.det0                = self.conf.get("det0", 225.0)
        self.klfd                = self.conf.get("klfd", -1.0)
        self.slope               = self.conf.get("slope", 0.0)
        self.amplifier_noise     = self.conf.get("amplifier_noise" , 0.1)
        self.adc_noise           = self.conf.get("adc_noise", 0.01)
        self.probe_scaling       = self.conf.get("probe_scaling", 0.83)
        self.cross_coupling_a    = complex(self.conf.get("cross_coupling_a", "1.12+0.1j"))
        self.cross_coupling_b    = complex(self.conf.get("cross_coupling_b", "-0.1-0.3j"))
        self.cross_coupling_c    = complex(self.conf.get("cross_coupling_c", "0.0+0.2j" ))
        self.cross_coupling_d    = complex(self.conf.get("cross_coupling_d", "0.3+1.0j" ))

        self.hbw_decay = 1.0

        self.abcd_cal = [1.0, 0.0, 0.0, 1.0]
        self.probe_cal = 1.0
        self.xy_cal = [1.0, 1.0]

        self.initialized = False

    def get_rf_traces_params(self):
        if not self.initialized:
            self.initialized = True
            self.samples = int(self.decay_stop * self.fs)
            self.time_trace = np.linspace(0, self.decay_stop, self.samples)
            self.probe_orig = np.zeros(self.samples, dtype=complex)
            self.vforw_orig = np.zeros(self.samples, dtype=complex)

            self.vforw_filling = self.vforw_flattop
            self.vforw_filling /= (1 - np.e **(-2.0 * np.pi *
                                               self.flattop_start *
                                               self.hbw))

            self.vforw_orig[:] = self.vforw_filling
            self.vforw_orig[self.time_trace > self.flattop_start] = self.vforw_flattop
            self.vforw_orig[self.time_trace > self.decay_start] = 0.0j
            self.vforw_orig += normal(scale=self.amplifier_noise, size=self.samples)

            def solve_cavity(t, x):
                x = x[0] + 1.0j * x[1]
                sigma = 2.0 * np.pi * (self.hbw + 1.0j * (self.det0 +
                                                          self.slope * t +
                                                          self.klfd * np.abs(x)**2))

                deriv = (-sigma * x +
                         4.0 * np.pi * self.hbw * np.interp(t, self.time_trace,
                                                            self.vforw_orig))

                return [np.real(deriv), np.imag(deriv)]

            r = ode(solve_cavity).set_integrator('lsoda')
            r.set_initial_value([0, 0], self.time_trace[0])
            self.probe_orig[0] = 0.0j

            for i, t in enumerate(self.time_trace[1:]):
                res = r.integrate(t)
                self.probe_orig[i] = res[0] + 1.0j * res[1]

            self.vrefl_orig = self.probe_orig - self.vforw_orig

        abcd_cross = np.array([[self.cross_coupling_a, self.cross_coupling_b],
                              [self.cross_coupling_c, self.cross_coupling_d]])

        abcd_inv = inv(abcd_cross)

        probe = np.array(self.probe_orig / self.probe_scaling)
        vforw = np.array(abcd_inv[0, 0] * self.vforw_orig + abcd_inv[0, 1] * self.vrefl_orig)
        vrefl = np.array(abcd_inv[1, 0] * self.vforw_orig + abcd_inv[1, 1] * self.vrefl_orig)

        probe *= self.probe_cal
        vforw *= self.xy_cal[0]
        vrefl *= self.xy_cal[1]

        probe += normal(scale=self.adc_noise, size=self.samples)
        vforw += normal(scale=self.adc_noise, size=self.samples)
        vrefl += normal(scale=self.adc_noise, size=self.samples)

        print("abcd total:", self.cross_coupling_a, self.cross_coupling_b,
                             self.cross_coupling_c, self.cross_coupling_d)

        return self.f0, self.fs, probe, vforw, vrefl, self.flattop_start, self.decay_start

    def get_abcd_scaling(self):
        return self.abcd_cal

    def set_abcd_scaling(self, a, b, c, d):
        self.abcd_cal = [a, b, c, d]

    def get_hbw_decay(self):
        return self.hbw_decay

    def set_hbw_decay(self, hbw):
        self.hbw_decay = hbw

    def get_xy_scaling(self):
        return self.xy_cal

    def set_xy_scaling(self, x, y):
        self.xy_cal = [x, y]

    def get_probe_amplitude_scaling(self):
        return self.probe_cal

    def set_probe_amplitude_scaling(self, scale):
        self.probe_cal = scale

    def get_cavity_voltage(self):
        volt = np.max(np.abs(self.probe_orig))
        return volt


