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
