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


