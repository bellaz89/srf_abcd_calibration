import numpy as np
from station import Station, AP2C, C2AP

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
        splitted_address = self.address.split("/")[:3]
        base_address = "/".join(splitted_address[:2]) + "/"
        system_name = splitted_address[-1].split(".", 1)[1] + "/"
        self.ctrl_address = base_address + "CTRL." + system_name
        config_address = base_address + "CONFIG." + system_name
        probe_address = base_address + "PROBE." + system_name
        vforw_address = base_address + "FORWARD." + system_name
        vrefl_address = base_address + "REFLECTED." + system_name

        self.ctrl_address = conf.get("ctrl_address", self.ctrl_address)
        config_address = conf.get("config_address", ctrl_address)
        probe_address = conf.get("probe_address", probe_address)
        vforw_address = conf.get("vforw_address", vforw_address)
        vrefl_address = conf.get("vrefl_address", vrefl_address)

        self.gradient_meter_address = conf.get("gradient_meter_address", None)
        self.gradient_meter_normalization = conf.get("gradient_meter_normalization", 1)

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

        self.ql_address = self.config_address + QL

        self.abcd = [complex(i) for i in [1, 0, 0, 1]]


    def get_rf_traces_params(self):

        # Get RF traces with the same macropulse
        probe_amp = pydoocs.read(self.probe_amp_address)
        macropulse = probe_amp["macropulse"]
        time_trace = 1.0e-6 * probe_amp["data"][:, 0]
        fs = 1.0/(time_trace[1] - time_trace[0])
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

        return (fs,
                AP2C(probe_amp, probe_pha),
                AP2C(vforw_amp, vforw_pha),
                AP2C(vrefl_amp, vrefl_pha),
                pulse_filling,
                pulse_filling + pulse_flattop)

    def get_abcd_scaling(self):
        return self.abcd

    def set_abcd_scaling(self, a, b, c, d):
        self.abcd = [a, b, c, d]

    def get_hbw_decay(self):
        ql = pydoocs.read(self.ql_address)["data"]
        return self.frequency / (2.0 * ql)

    def set_hbw_decay(seĺf, hbw):
        ql = self.frequency / (2.0 * hbw)
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


class DesyDoocsMCAVStation(Station):
    def __init__(self, name, conf):
        super(DesyDoocsMCAVStation, self).__init__(name, conf)

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

        self.hbw = 130

    def get_hbw_decay(self):
        return self.hbw

    def set_hbw_decay(seĺf, hbw):
        self.hbw = hbw


