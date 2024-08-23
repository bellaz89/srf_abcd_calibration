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
try:
    import pydoocs
except:
    pydoocs = None

class DesyDoocsSCAVStation(Station):
    @staticmethod
    def loadable():
        return pydoocs is not None

    def __init__(self, name, conf):
        super(DesyDoocsSCAVStation, self).__init__(name, conf)
        self.address = conf["address"]
        self.splitted_address = self.address.split("/")[:3]
        self.address = "/".join(self.splitted_address) + "/"
        self.base_address = "/".join(self.splitted_address[:2]) + "/"
        self.system_name = self.splitted_address[-1].split(".", 1)[1]
        self.ctrl_address = self.base_address + "CTRL." + self.system_name + "/"
        self.config_address = self.base_address + "CONFIG." + self.system_name + "/"
        probe_address = self.base_address + "PROBE." + self.system_name + "/"
        vforw_address = self.base_address + "FORWARD." + self.system_name + "/"
        vrefl_address = self.base_address + "REFLECTED." + self.system_name + "/"

        self.ctrl_address = conf.get("ctrl_address", self.ctrl_address)
        self.config_address = conf.get("config_address", self.config_address)
        probe_address = conf.get("probe_address", probe_address)
        vforw_address = conf.get("vforw_address", vforw_address)
        vrefl_address = conf.get("vrefl_address", vrefl_address)

        self.gradient_meter_address = conf.get("gradient_meter_address", None)
        self.gradient_meter_normalization = conf.get("gradient_meter_normalization", 1.0)

        self.probe_amp_address = probe_address + "AMPL"
        self.probe_pha_address = probe_address + "PHASE"
        self.vforw_amp_address = vforw_address + "AMPL"
        self.vforw_pha_address = vforw_address + "PHASE"
        self.vrefl_amp_address = vrefl_address + "AMPL"
        self.vrefl_pha_address = vrefl_address + "PHASE"
        self.probe_cal_amp_address = probe_address + "CAL_SCA"
        self.probe_cal_pha_address = probe_address + "CAL_ROT"
        self.vforw_cal_amp_address = vforw_address + "CAL_SCA"
        self.vforw_cal_pha_address = vforw_address + "CAL_ROT"
        self.vrefl_cal_amp_address = vrefl_address + "CAL_SCA"
        self.vrefl_cal_pha_address = vrefl_address + "CAL_ROT"

        self.pulse_delay_address = self.ctrl_address + "PULSE_DELAY"
        self.pulse_filling_address = self.ctrl_address + "PULSE_FILLING"
        self.pulse_flattop_address = self.ctrl_address + "PULSE_FLATTOP"

        self.decoupling_a_re_address = self.ctrl_address + "DECOUPLING.A_RE"
        self.decoupling_a_im_address = self.ctrl_address + "DECOUPLING.A_IM"
        self.decoupling_b_re_address = self.ctrl_address + "DECOUPLING.B_RE"
        self.decoupling_b_im_address = self.ctrl_address + "DECOUPLING.B_IM"
        self.decoupling_c_re_address = self.ctrl_address + "DECOUPLING.C_RE"
        self.decoupling_c_im_address = self.ctrl_address + "DECOUPLING.C_IM"
        self.decoupling_d_re_address = self.ctrl_address + "DECOUPLING.D_RE"
        self.decoupling_d_im_address = self.ctrl_address + "DECOUPLING.D_IM"

        self.ql_address = self.config_address + "QL"
        self.f0_address = self.config_address + "F0"
        self.fs_address = self.config_address + "FS"

    def get_rf_traces_params(self):
        # Get RF traces with the same macropulse
        f0 = pydoocs.read(self.f0_address)["data"] * 1e6
        fs = pydoocs.read(self.fs_address)["data"] * 1e6
        probe_amp = pydoocs.read(self.probe_amp_address)
        macropulse = probe_amp["macropulse"]
        probe_amp = probe_amp["data"][:, 1]
        probe_pha = pydoocs.read(self.probe_pha_address, macropulse=macropulse)["data"][:, 1]
        vforw_amp = pydoocs.read(self.vforw_amp_address, macropulse=macropulse)["data"][:, 1]
        vforw_pha = pydoocs.read(self.vforw_pha_address, macropulse=macropulse)["data"][:, 1]
        vrefl_amp = pydoocs.read(self.vrefl_amp_address, macropulse=macropulse)["data"][:, 1]
        vrefl_pha = pydoocs.read(self.vrefl_pha_address, macropulse=macropulse)["data"][:, 1]

        pulse_delay = 1.0e-6 * pydoocs.read(self.pulse_delay_address)["data"]
        pulse_filling = 1.0e-6 * pydoocs.read(self.pulse_filling_address)["data"]
        pulse_flattop = 1.0e-6 * pydoocs.read(self.pulse_flattop_address)["data"]

        rf_pulse_time = time_trace > pulse_delay
        probe_amp = probe_amp[rf_pulse_time]
        probe_pha = probe_pha[rf_pulse_time]
        vforw_amp = vforw_amp[rf_pulse_time]
        vforw_pha = vforw_pha[rf_pulse_time]
        vrefl_amp = vrefl_amp[rf_pulse_time]
        vrefl_pha = vrefl_pha[rf_pulse_time]

        return (f0,
                fs,
                AP2C(probe_amp, probe_pha),
                AP2C(vforw_amp, vforw_pha),
                AP2C(vrefl_amp, vrefl_pha),
                pulse_filling,
                pulse_filling + pulse_flattop)

    def get_abcd_scaling(self):
        a = (pydoocs.read(self.decoupling_a_re_address)["data"][0] +
             pydoocs.read(self.decoupling_a_im_address)["data"][0] * 1.0j)
        b = (pydoocs.read(self.decoupling_b_re_address)["data"][0] +
             pydoocs.read(self.decoupling_b_im_address)["data"][0] * 1.0j)
        c = (pydoocs.read(self.decoupling_c_re_address)["data"][0] +
             pydoocs.read(self.decoupling_c_im_address)["data"][0] * 1.0j)
        d = (pydoocs.read(self.decoupling_d_re_address)["data"][0] +
             pydoocs.read(self.decoupling_d_im_address)["data"][0] * 1.0j)

        return [a, b, c, d]

    def set_abcd_scaling(self, a, b, c, d):
        pydoocs.write(self.decoupling_a_re_address, [np.real(a)] + [0.0] * 15)
        pydoocs.write(self.decoupling_a_im_address, [np.imag(a)] + [0.0] * 15)
        pydoocs.write(self.decoupling_b_re_address, [np.real(b)] + [0.0] * 15)
        pydoocs.write(self.decoupling_b_im_address, [np.imag(b)] + [0.0] * 15)
        pydoocs.write(self.decoupling_c_re_address, [np.real(c)] + [0.0] * 15)
        pydoocs.write(self.decoupling_c_im_address, [np.imag(c)] + [0.0] * 15)
        pydoocs.write(self.decoupling_d_re_address, [np.real(d)] + [0.0] * 15)
        pydoocs.write(self.decoupling_d_im_address, [np.imag(d)] + [0.0] * 15)

    def get_hbw_decay(self):
        ql = pydoocs.read(self.ql_address)["data"]
        f0 = pydoocs.read(self.f0_address)["data"] * 1e6
        return  f0 / (2.0 * ql)

    def set_hbw_decay(seĺf, hbw):
        f0 = pydoocs.read(self.f0_address)["data"] * 1e6
        ql = f0 / (2.0 * hbw)
        pydoocs.write(self.ql_address, ql)

    def get_xy_scaling(self):
        x_amp = pydoocs.write(self.vforw_cal_amp_address)
        x_pha = pydoocs.write(self.vforw_cal_pha_address)
        y_amp = pydoocs.write(self.vrefl_cal_amp_address)
        y_pha = pydoocs.write(self.vrefl_cal_pha_address)
        return (AP2C(x_amp, x_pha), AP2C(y_amp, y_pha))

    def set_xy_scaling(self, x, y):
        (x_amp, x_pha) = C2AP(x)
        (y_amp, y_pha) = C2AP(y)
        pydoocs.write(self.vforw_cal_amp_address, x_amp)
        pydoocs.write(self.vforw_cal_pha_address, x_pha)
        pydoocs.write(self.vrefl_cal_amp_address, y_amp)
        pydoocs.write(self.vrefl_cal_pha_address, y_pha)

    def get_probe_amplitude_scaling(self):
        return pydoocs.read(self.probe_cal_amp_addres)["data"]

    def set_probe_amplitude_scaling(self, scale):
        pydoocs.write(self.probe_cal_amp_address, scale)

    def get_cavity_voltage(self):
        if self.gradient_meter_address:
            gradient = self.gradient_meter_normalization
            gradient *= pydoocs.read(self.gradient_meter_address)["address"]
            return gradient
        else:
            return None


class DesyDoocsMCAVStation(DesyDoocsSCAVStation):
    def __init__(self, name, conf):
        super(DesyDoocsMCAVStation, self).__init__(name, conf)

        self.system_name = self.splitted_address[-1].split(".", 2)[2]

        self.main_address = self.base_address + "MAIN." + self.system_name + "/"
        self.ctrl_address = self.base_address + "CTRL." + self.system_name + "/"
        self.config_address = self.base_address + "CONFIG." + self.system_name + "/"

        self.main_address = self.conf.get("main_address", self.main_address)
        self.ctrl_address = conf.get("ctrl_address", self.ctrl_address)
        self.config_address = conf.get("config_address", self.config_address)
        probe_address = self.address
        vforw_address = self.address
        vrefl_address = self.address

        self.probe_amp_address = self.address + "PROBE.AMPL"
        self.probe_pha_address = self.address + "PROBE.PHASE"
        self.vforw_amp_address = self.address + "VFORW.AMPL"
        self.vforw_pha_address = self.address + "VFORW.PHASE"
        self.vrefl_amp_address = self.address + "VREFL.AMPL"
        self.vrefl_pha_address = self.address + "VREFL.PHASE"
        self.probe_cal_amp_address = self.address + "PROBE.CAL_SCA"
        self.probe_cal_pha_address = self.address + "PROBE.CAL_ROT"
        self.vforw_cal_amp_address = self.address + "VFORW.CAL_SCA"
        self.vforw_cal_pha_address = self.address + "VFORW.CAL_ROT"
        self.vrefl_cal_amp_address = self.address + "VREFL.CAL_SCA"
        self.vrefl_cal_pha_address = self.address + "VREFL.CAL_ROT"

        self.pulse_delay_address = self.ctrl_address + "PULSE_DELAY"
        self.pulse_filling_address = self.ctrl_address + "PULSE_FILLING"
        self.pulse_flattop_address = self.ctrl_address + "PULSE_FLATTOP"

        self.decoupling_a_re_address = self.address + "DECOUPLING.A_RE"
        self.decoupling_a_im_address = self.address + "DECOUPLING.A_IM"
        self.decoupling_b_re_address = self.address + "DECOUPLING.B_RE"
        self.decoupling_b_im_address = self.address + "DECOUPLING.B_IM"
        self.decoupling_c_re_address = self.address + "DECOUPLING.C_RE"
        self.decoupling_c_im_address = self.address + "DECOUPLING.C_IM"
        self.decoupling_d_re_address = self.address + "DECOUPLING.D_RE"
        self.decoupling_d_im_address = self.address + "DECOUPLING.D_IM"

        self.f0_address = self.main_address + "F0"
        self.fs_address = self.main_address + "FS"
        self.hbw = 130
        self.abcd = [1.0, 0.0, 0.0, 1.0]

    def get_hbw_decay(self):
        return self.hbw

    def set_hbw_decay(seĺf, hbw):
        self.hbw = hbw

    def get_abcd_scaling(self):
        return self.abcd

    def set_abcd_scaling(self, a, b, c, d):
        self.abcd = [a, b, c, d]


