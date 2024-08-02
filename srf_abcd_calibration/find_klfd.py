import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def find_klfd(time_trace, probe_cmplx, detuning):
    probe_amp_sq = np.abs(probe_cmplx)**2

    def klfd_square(t, det0, klfd, det_slope):
        return det0 + klfd * np.interp(t, time_trace, probe_amp_sq) + det_slope * t

    popt, pcov = curve_fit(klfd_square, time_trace, detuning, p0 = [0, -1, 0])
    return (popt[0], popt[1], popt[2])


