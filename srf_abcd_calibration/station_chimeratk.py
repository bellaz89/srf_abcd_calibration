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

import os
from itertools import cycle
from .station import Station

try:
    import deviceaccess as da
except:
    da = None

class ChimeraTKStation(Station):
    @staticmethod
    def loadable():
        return da is not None

    def __init__(self, name, conf):
        super(DesyDoocsSCAVStation, self).__init__(name, conf)
        self.dmap_path = conf["dmap_path"]
        self.device_name = conf["device_name"]

        self.gradient_meter_address = conf.get("gradient_meter_address", None)
        self.gradient_meter_normalization = conf.get("gradient_meter_normalization", 1.0)

        self.probe_amp_address = conf["probe_amp_address"]
        self.probe_pha_address = conf["probe_pha_address"]
        self.vforw_amp_address = conf["vforw_amp_address"]
        self.vforw_pha_address = conf["vforw_pha_address"]
        self.vrefl_amp_address = conf["vrefl_amp_address"]
        self.vrefl_pha_address = conf["vrefl_pha_address"]
        self.probe_cal_amp_address = conf["probe_cal_amp_address"]
        self.probe_cal_pha_address = conf["probe_cal_pha_address"]
        self.vforw_cal_amp_address = conf["vforw_cal_amp_address"]
        self.vforw_cal_pha_address = conf["vforw_cal_pha_address"]
        self.vrefl_cal_amp_address = conf["vrefl_cal_amp_address"]
        self.vrefl_cal_pha_address = conf["vrefl_cal_pha_address"]
        self.pulse_delay_address = conf["pulse_delay_address"]
        self.pulse_filling_address = conf["pulse_filling_address"]
        self.pulse_flattop_address = conf["pulse_flattop_address"]
        self.decoupling_a_re_address = conf["decoupling_a_re_address"]
        self.decoupling_a_im_address = conf["decoupling_a_im_address"]
        self.decoupling_b_re_address = conf["decoupling_b_re_address"]
        self.decoupling_b_im_address = conf["decoupling_b_im_address"]
        self.decoupling_c_re_address = conf["decoupling_c_re_address"]
        self.decoupling_c_im_address = conf["decoupling_c_im_address"]
        self.decoupling_d_re_address = conf["decoupling_d_re_address"]
        self.decoupling_d_im_address = conf["decoupling_d_im_address"]

        self.ql_address = conf["ql_address"]
        self.f0_address = conf["f0_address"]
        self.fs_address = conf["fs_address"]

    def set_dmap(self):
        dmap_path = os.path.split(self.dmap_path)
        self.cwd = os.getcwd()
        os.chdir(dmap_path[0])
        da.setDMapFilePath(dmap_path[1])

    def unset_dmap(self):
        os.chdir(self.cwd)

    def get_rf_traces_params(self):
        # Get RF traces with the same macropulse
        set_dmap()

        f0 = None
        fs = None
        probe = None
        vforw = None
        vrefl = None
        pulse_delay = None
        pulse_filling = None
        pulse_flattop = None

        with da.Device(self.device_name) as dev:
            dev.activateAsyncRead()
            probe_amp_acc = dev.getOneDRegisterAccessor(np.float32, probe_amp_address)
            probe_pha_acc = dev.getOneDRegisterAccessor(np.float32, probe_pha_address)
            vforw_amp_acc = dev.getOneDRegisterAccessor(np.float32, vforw_amp_address)
            vforw_pha_acc = dev.getOneDRegisterAccessor(np.float32, vforw_pha_address)
            vrefl_amp_acc = dev.getOneDRegisterAccessor(np.float32, vrefl_amp_address)
            vrefl_pha_acc = dev.getOneDRegisterAccessor(np.float32, vrefl_pha_address)

            channels = [("probe_amp", probe_amp_acc),
                        ("probe_pha", probe_pha_acc),
                        ("vforw_amp", vforw_amp_acc),
                        ("vforw_pha", vforw_pha_acc),
                        ("vrefl_amp", vrefl_amp_acc),
                        ("vrefl_pha", vrefl_pha_acc)]

            read_channels = dict()
            version_number = None

            for name, acc in cycle(channels):
                acc.read()
                version_number = acc.getVersionNumber()
                ch = read_channels.get(version_number, dict())
                read_channels[version_number] = ch
                ch[name] = np.array(acc)

                if len(ch) == 6:
                    probe = AP2C(ch["probe_amp"], ch["probe_pha"])
                    vforw = AP2C(ch["vforw_amp"], ch["vforw_pha"])
                    vrefl = AP2C(ch["vrefl_amp"], ch["vrefl_pha"])
                    break

            f0_acc = dev.getScalarRegisterAccessor(np.float32, self.f0_address)
            f0_acc.read()
            f0 = f0_acc[0] * 1.0e6

            fs_acc = dev.getScalarRegisterAccessor(np.float32, self.fs_address)
            fs_acc.read()
            fs = fs_acc[0]  * 1.0e6

            pulse_delay_acc = dev.getScalarRegisterAccessor(np.float32, self.pulse_delay_address)
            pulse_delay_acc.read()
            pulse_delay = pulse_delay_acc * 1.0e-6

            pulse_filling_acc = dev.getScalarRegisterAccessor(np.float32, self.pulse_filling_address)
            pulse_filling_acc.read()
            pulse_filling = pulse_filling_acc * 1.0e-6

            pulse_flattop_acc = dev.getScalarRegisterAccessor(np.float32, self.pulse_flattop_address)
            pulse_flattop_acc.read()
            pulse_flattop = pulse_flattop_acc * 1.0e-6

        trace_time = probe.shape[0] / fs
        time_trace = np.linspace(0, trace_time, probe.shape[0])

        rf_pulse_time = time_trace > pulse_delay

        probe = probe[rf_pulse_time]
        vforw = vforw[rf_pulse_time]
        vrefl = vrefl[rf_pulse_time]

        pulse_decay += pulse_flattop

        unset_dmap()

        return (f0,
                fs,
                probe,
                vforw,
                vrefl,
                pulse_flattop,
                pulse_decay)

    def get_abcd_scaling(self):
        set_dmap()

        abcd = [0.0j, 0.0j, 0.0j, 0.0j]

        with da.Device(self.device_name) as dev:
            a_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_a_re_address)
            a_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_a_im_address)
            b_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_b_re_address)
            b_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_b_im_address)
            c_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_c_re_address)
            c_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_c_im_address)
            d_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_d_re_address)
            d_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_d_im_address)
            a_re_acc.read()
            a_im_acc.read()
            b_re_acc.read()
            b_im_acc.read()
            c_re_acc.read()
            c_im_acc.read()
            d_re_acc.read()
            d_im_acc.read()

            abcd[0] = a_re_acc[0] + a_im_acc[0]
            abcd[1] = b_re_acc[1] + b_im_acc[1]
            abcd[2] = c_re_acc[2] + c_im_acc[2]
            abcd[3] = d_re_acc[3] + d_im_acc[3]

        unset_dmap()

        return abcd

    def set_abcd_scaling(self, a, b, c, d):
        set_dmap()

        with da.Device(self.device_name) as dev:
            a_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_a_re_address)
            a_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_a_im_address)
            b_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_b_re_address)
            b_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_b_im_address)
            c_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_c_re_address)
            c_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_c_im_address)
            d_re_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_d_re_address)
            d_im_acc = dev.getOneDRegisterAccessor(np.float32, self.decoupling_d_im_address)

            a_re_acc.read()
            a_im_acc.read()
            b_re_acc.read()
            b_im_acc.read()
            c_re_acc.read()
            c_im_acc.read()
            d_re_acc.read()
            d_im_acc.read()

            a_re_acc[0] = np.real(a)
            a_im_acc[0] = np.imag(a)
            b_re_acc[0] = np.real(b)
            b_im_acc[0] = np.imag(b)
            c_re_acc[0] = np.real(c)
            c_im_acc[0] = np.imag(c)
            d_re_acc[0] = np.real(d)
            d_im_acc[0] = np.imag(d)

            a_re_acc.write()
            a_im_acc.write()
            b_re_acc.write()
            b_im_acc.write()
            c_re_acc.write()
            c_im_acc.write()
            d_re_acc.write()
            d_im_acc.write()

        unset_dmap()

    def get_hbw_decay(self):
        set_dmap()

        f0 = None
        ql = None

        with da.Device(self.device_name) as dev:
            ql_acc = dev.getOneDRegisterAccessor(np.float32, self.ql_address)
            ql_acc.read()
            ql = ql_acc[0]
            f0_acc = dev.getScalarRegisterAccessor(np.float32, self.f0_address)
            f0_acc.read()
            f0 = f0_acc[0] * 1.0e6

        unset_dmap()

        return f0 / (2.0 * ql)

    def set_hbw_decay(seĺf, hbw):
        set_dmap()

        with da.Device(self.device_name) as dev:
            f0_acc = dev.getScalarRegisterAccessor(np.float32, self.f0_address)
            f0_acc.read()
            f0 = f0_acc[0] * 1.0e6
            ql_acc = dev.getOneDRegisterAccessor(np.float32, self.ql_address)
            ql_acc.read()
            ql_acc.set(f0 / (2.0 * hbw))
            ql_acc.write()

        unset_dmap()

    def get_xy_scaling(self):
        set_dmap()

        x = None
        y = None

        with da.Device(self.device_name) as dev:
            vforw_cal_amp_acc = dev.getScalarRegisterAccessor(np.float, self.vforw_cal_amp_address)
            vforw_cal_pha_acc = dev.getScalarRegisterAccessor(np.float, self.vforw_cal_pha_address)
            vrefl_cal_amp_acc = dev.getScalarRegisterAccessor(np.float, self.vrefl_cal_amp_address)
            vrefl_cal_pha_acc = dev.getScalarRegisterAccessor(np.float, self.vrefl_cal_pha_address)
            vforw_cal_amp_acc.read()
            vforw_cal_pha_acc.read()
            vrefl_cal_amp_acc.read()
            vrefl_cal_pha_acc.read()
            x = AP2C(vforw_cal_amp_acc.get(), vforw_cal_pha_acc.get())
            y = AP2C(vrefl_cal_amp_acc.get(), vrefl_cal_pha_acc.get())

        unset_dmap()
        return (x, y)

    def set_xy_scaling(self, x, y):
        set_dmap()

        (x_amp, x_pha) = C2AP(x)
        (y_amp, y_pha) = C2AP(y)

        with da.Device(self.device_name) as dev:
            vforw_cal_amp_acc = dev.getScalarRegisterAccessor(np.float, self.vforw_cal_amp_address)
            vforw_cal_pha_acc = dev.getScalarRegisterAccessor(np.float, self.vforw_cal_pha_address)
            vrefl_cal_amp_acc = dev.getScalarRegisterAccessor(np.float, self.vrefl_cal_amp_address)
            vrefl_cal_pha_acc = dev.getScalarRegisterAccessor(np.float, self.vrefl_cal_pha_address)
            vforw_cal_amp_acc.read()
            vforw_cal_pha_acc.read()
            vrefl_cal_amp_acc.read()
            vrefl_cal_pha_acc.read()
            vforw_cal_amp_acc.set(x_amp)
            vforw_cal_pha_acc.set(x_pha)
            vrefl_cal_amp_acc.set(y_amp)
            vrefl_cal_pha_acc.set(y_pha)
            vforw_cal_amp_acc.write()
            vforw_cal_pha_acc.write()
            vrefl_cal_amp_acc.write()
            vrefl_cal_pha_acc.write()

        unset_dmap()

    def get_probe_amplitude_scaling(self):
        set_dmap()

        scale = None

        with da.Device(self.device_name) as dev:
            probe_cal_amp_acc = dev.getScalarRegisterAccessor(np.float, self.probe_cal_amp_address)
            probe_cal_pha_acc = dev.getScalarRegisterAccessor(np.float, self.probe_cal_pha_address)
            probe_cal_amp_acc.read()
            probe_cal_pha_acc.read()
            scale = AP2C(probe_cal_amp_acc.get(), probe_cal_pha_acc.get())

        unset_dmap()
        return scale

    def set_probe_amplitude_scaling(self, scale):
        set_dmap()

        (sca_amp, sca_pha) = C2AP(scale)

        with da.Device(self.device_name) as dev:
            probe_cal_amp_acc = dev.getScalarRegisterAccessor(np.float, self.probe_cal_amp_address)
            probe_cal_pha_acc = dev.getScalarRegisterAccessor(np.float, self.probe_cal_pha_address)
            probe_cal_amp_acc.read()
            probe_cal_pha_acc.read()
            probe_cal_amp_acc.set(sca_amp)
            probe_cal_pha_acc.set(sca_pha)
            probe_cal_amp_acc.write()
            probe_cal_pha_acc.write()

        unset_dmap()

    def get_cavity_voltage(self):
        gradient = None

        if self.gradient_meter_address:
            set_dmap()
            with da.Device(self.device_name) as dev:
                gradient = self.gradient_meter_normalization
                gradient_acc = dev.getScalarRegisterAccessor(np.float, self.gradient_meter_address)
                gradient_acc.read()
                gradient *= gradient_acc.get()

            unset_dmap()

        return gradient
