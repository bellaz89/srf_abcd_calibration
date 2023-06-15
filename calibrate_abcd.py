import pydoocs
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, least_squares
from scipy.signal import savgol_filter
from time import sleep
import warnings

FRAC_OPT = 0.02
ITER_OPT = 50

def AP2C(amplitude, phase):
    return amplitude * np.exp(1.0j * phase * np.pi / 180)

def C2AP(cmplx):
    return np.abs(cmplx), np.angle(cmplx, deg=True)

def C2REIM(cmplx):
    reim = np.zeros((cmplx.shape[0], 2)) 
    return (np.real(cmplx), np.imag(cmplx))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_regions_duration(delay_addr, filling_addr, flattop_addr):
    return (pydoocs.read(delay_addr)["data"],
            pydoocs.read(filling_addr)["data"],
            pydoocs.read(flattop_addr)["data"])

def get_range_idx(time_trace, decay_length, filling_delay, delay_addr, filling_addr, flattop_addr):
    (delay, filling, flattop) = get_regions_duration(delay_addr, filling_addr, flattop_addr)
    start = delay + filling_delay
    stop =  delay + filling + flattop + decay_length
    return (find_nearest(time_trace, start), find_nearest(time_trace, stop))

def get_pulse_range_idx(time_trace, decay_length, filling_delay, delay_addr, filling_addr, flattop_addr):
    (delay, filling, flattop) = get_regions_duration(delay_addr, filling_addr, flattop_addr)
    start = delay
    stop =  delay + filling
    return (find_nearest(time_trace, start), find_nearest(time_trace, stop))

def get_decay_range_idx(time_trace, decay_length, delay_decay, delay_addr, filling_addr, flattop_addr):
    (delay, filling, flattop) = get_regions_duration(delay_addr, filling_addr, flattop_addr)
    start = delay + filling + flattop + delay_decay
    stop = delay + filling + flattop + decay_length
    return (find_nearest(time_trace, start), find_nearest(time_trace, stop))

def find_QL(f0, probe_amp_addr, threshold, delay_decay, delay_addr, filling_addr, flattop_addr):
    probe = pydoocs.read(probe_amp_addr)["data"]
    (delay, filling, flattop) = get_regions_duration(delay_addr, filling_addr, flattop_addr)

    def exponential_decay(t, a, b):
        return a * np.exp(-t * np.pi * f0 / b)

    time_trace = probe[:, 0]
    amplitude = probe[:, 1]

    decay_start_idx = get_decay_range_idx(time_trace, 0, delay_decay, delay_addr, filling_addr, flattop_addr)[0]

    time_trace = time_trace[decay_start_idx:]
    time_trace = time_trace - time_trace[0]
    amplitude = amplitude[decay_start_idx:]

    decay_stop_idx = find_nearest(amplitude, amplitude[0] * threshold)
    time_trace = time_trace[:decay_stop_idx]
    amplitude = amplitude[:decay_stop_idx]

    popt, pcov = curve_fit(exponential_decay, 
                           time_trace / 1e6, amplitude,
                           p0 = [amplitude[0], 1e7])

    return popt[1]

def get_traces_cmplx(probe_amp_addr, probe_pha_addr, 
                     forward_amp_addr, forward_pha_addr, 
                     reflected_amp_addr, reflected_pha_addr):

    probe_cmplx = []
    forward_cmplx = []
    reflected_cmplx = []

    probe_amp = pydoocs.read(probe_amp_addr)
    macropulse = probe_amp["macropulse"]
    time_trace = probe_amp["data"][:, 0]
    probe_amp = probe_amp["data"][:, 1]
    probe_pha = pydoocs.read(probe_pha_addr, macropulse=macropulse)["data"][:, 1]
    forward_amp = pydoocs.read(forward_amp_addr, macropulse=macropulse)["data"][:, 1]
    forward_pha = pydoocs.read(forward_pha_addr, macropulse=macropulse)["data"][:, 1]
    reflected_amp = pydoocs.read(reflected_amp_addr, macropulse=macropulse)["data"][:, 1]
    reflected_pha = pydoocs.read(reflected_pha_addr, macropulse=macropulse)["data"][:, 1]

    probe_cmplx = AP2C(probe_amp, probe_pha)
    forward_cmplx = AP2C(forward_amp, forward_pha)
    reflected_cmplx = AP2C(reflected_amp, reflected_pha)

    return (time_trace, 
            probe_cmplx,
            forward_cmplx,
            reflected_cmplx)


def calibrate_xy(probe_cmplx, forward_cmplx, reflected_cmplx):
    cols = probe_cmplx.shape[0]
    A = np.zeros((cols, 2), dtype=complex)
    A[:, 0] = forward_cmplx
    A[:, 1] = reflected_cmplx

    return np.linalg.lstsq(A, probe_cmplx, rcond=None)[0]

def calibrate_abcd(hbw,
                   probe_cmplx, forward_cmplx, reflected_cmplx, dU, U,
                   QL_weight = 1):

    (probe_re, probe_im)         = C2REIM(probe_cmplx)
    (forward_re, forward_im)     = C2REIM(forward_cmplx)
    (reflected_re, reflected_im) = C2REIM(reflected_cmplx)

    zeros = np.zeros_like(probe_re)

    A_re = [forward_re, -forward_im, 
            reflected_re, -reflected_im] * 2

    A_im = [forward_im,  forward_re, 
            reflected_im, -reflected_re] * 2 

    A_Q  = ([4 * QL_weight * (  probe_re * forward_re   + probe_im * forward_im), 
             4 * QL_weight * (- probe_re * forward_im   + probe_im * forward_re),
             4 * QL_weight * (  probe_re * reflected_re + probe_im * reflected_im),
             4 * QL_weight * (- probe_re * reflected_im + probe_im * reflected_re)] + 
            [zeros] * 4)

    A_re = np.column_stack(A_re)
    A_im = np.column_stack(A_im)
    A_Q  = np.column_stack(A_Q)

    A = np.vstack((A_re, 
                   A_im,     
                   A_Q))

    b_re = probe_re
    b_im = probe_im
    b_Q  = QL_weight * (dU/ (2 * np.pi * hbw) + 2 * U)

    b = np.concatenate((b_re, 
                        b_im, 
                        b_Q))

    #b = np.reshape(b, (b.shape[0], 1))
    #x = np.linalg.lstsq(A, b, rcond=None)[0]

    def opt_fun(x):
        vforw_c_cmplx = forward_cmplx * (x[0] + 1.0j * x[1]) + reflected_cmplx * (x[2] + 1.0j * x[3])
        vrefl_c_cmplx = forward_cmplx * (x[4] + 1.0j * x[5]) + reflected_cmplx * (x[6] + 1.0j * x[7])

        diff_U = np.abs(vforw_c_cmplx)**2 - np.abs(vrefl_c_cmplx)**2
        #return np.concatenate((np.dot(A, x) - b, dU / (2 * np.pi * hbw) - diff_U))
        return np.concatenate((np.dot(A, x) - b, 
                               dU / (2 * np.pi * hbw) - 2*diff_U))

    x = least_squares(opt_fun, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).x

    return (x[0] + 1.0j * x[1], 
            x[2] + 1.0j * x[3], 
            x[4] + 1.0j * x[5], 
            x[6] + 1.0j * x[7])


def compute_bwdet(time_trace, hbw, probe, forward, reflected):

    dt = (time_trace[1] - time_trace[0]) * 1e-6
    probe_deriv = np.gradient(probe) / dt

    A2 = np.abs(probe) ** 2

    rt = 2*hbw*forward - probe_deriv / (2.0 * np.pi)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bwdet = probe * np.conj(rt) / A2
        return (2 * np.real(bwdet), -np.imag(bwdet))


def calculate_abcd(f0, delay_decay, filling_delay, 
                   probe_amp_addr, probe_pha_addr, 
                   vforw_amp_addr, vforw_pha_addr, 
                   vrefl_amp_addr, vrefl_pha_addr, 
                   delay_addr, filling_addr, flattop_addr, 
                   threshold, flatten_s):

    FILTER_RANGE = 101

    QL = find_QL(f0, probe_amp_addr, threshold, delay_decay, delay_addr, filling_addr, flattop_addr)
    hbw = 0.5 * f0 / QL
    decay_length = np.pi * QL / f0 * 1e6
    (time_trace, probe, vforw_orig, vrefl_orig) = get_traces_cmplx(probe_amp_addr, probe_pha_addr,
                                                                   vforw_amp_addr, vforw_pha_addr,
                                                                   vrefl_amp_addr, vrefl_pha_addr)

    max_amp = np.max(np.abs(probe))
    
    probe /= max_amp
    vforw_orig /= max_amp
    vrefl_orig /= max_amp

    (start_idx, stop_idx) = get_range_idx(time_trace, decay_length, filling_delay,
                                          delay_addr, filling_addr, flattop_addr)
    
    xy = calibrate_xy(probe[start_idx:stop_idx],
                      vforw_orig[start_idx:stop_idx],
                      vrefl_orig[start_idx:stop_idx])

    x = xy[0]
    y = xy[1]

    vforw_xy = vforw_orig * x
    vrefl_xy = vrefl_orig * y

    (start_pulse_idx, stop_pulse_idx) = get_pulse_range_idx(time_trace, 
                                                            decay_length, 
                                                            filling_delay,
                                                            delay_addr, 
                                                            filling_addr, 
                                                            flattop_addr)

    (start_decay_idx, stop_decay_idx) = get_decay_range_idx(time_trace, 
                                                            decay_length, 
                                                            filling_delay,
                                                            delay_addr, 
                                                            filling_addr, 
                                                            flattop_addr)

    dt = (time_trace[1] - time_trace[0]) * 1e-6
    U = np.abs(probe)**2
    dU = savgol_filter(U, FILTER_RANGE, 3, deriv=1, delta=dt)

    start_pulse_idx += FILTER_RANGE
    stop_pulse_idx -= FILTER_RANGE 
    start_decay_idx += FILTER_RANGE
    stop_decay_idx -= FILTER_RANGE 

    probe_r = np.concatenate((probe[start_pulse_idx:stop_pulse_idx], probe[start_decay_idx:stop_decay_idx]))
    vforw_r = np.concatenate((vforw_xy[start_pulse_idx:stop_pulse_idx], vforw_xy[start_decay_idx:stop_decay_idx]))
    vrefl_r = np.concatenate((vrefl_xy[start_pulse_idx:stop_pulse_idx], vrefl_xy[start_decay_idx:stop_decay_idx]))
    U_r = np.concatenate((U[start_pulse_idx:stop_pulse_idx], U[start_decay_idx:stop_decay_idx]))
    dU_r = np.concatenate((dU[start_pulse_idx:stop_pulse_idx], dU[start_decay_idx:stop_decay_idx]))

    print(time_trace[start_pulse_idx], time_trace[stop_pulse_idx])
    print(time_trace[start_decay_idx], time_trace[stop_decay_idx])

    (a, b, c, d) = calibrate_abcd(hbw, probe_r, vforw_r, vrefl_r, dU_r, U_r, QL_weight = 1)

    vforw_abcd = a * vforw_xy + b * vrefl_xy 
    vrefl_abcd = c * vforw_xy + d * vrefl_xy

    (bandwidth_xy, detuning_xy) = compute_bwdet(time_trace, hbw, probe, vforw_xy, vrefl_xy)
    (bandwidth_abcd, detuning_abcd) = compute_bwdet(time_trace, hbw, probe, vforw_abcd, vrefl_abcd)


    bandwidth_xy[np.isnan(bandwidth_xy)] = 0.0
    bandwidth_abcd[np.isnan(bandwidth_abcd)] = 0.0
    detuning_xy[np.isnan(detuning_xy)] = 0.0 
    detuning_abcd[np.isnan(detuning_abcd)] = 0.0 

    if int(flatten_s/dt) * 2 + 1 > 3:
        bandwidth_xy  = savgol_filter(bandwidth_xy,  int(flatten_s/dt) * 2 + 1, 3)
        bandwidth_abcd  = savgol_filter(bandwidth_abcd,  int(flatten_s/dt) * 2 + 1, 3)
        detuning_xy = savgol_filter(detuning_xy, int(flatten_s/dt) * 2 + 1, 3)
        detuning_abcd = savgol_filter(detuning_abcd, int(flatten_s/dt) * 2 + 1, 3)

    result = dict()

    result["time_trace"] = time_trace
    result["QL"] = QL
    result["bw"] = 2*hbw
    result["decay_length"] = decay_length
    result["xy"] = (x, y)
    result["abcd"] = (a, b, c, d)
    result["probe"] = probe * max_amp
    result["vforw_orig"] = vforw_orig * max_amp
    result["vrefl_orig"] = vrefl_orig * max_amp
    result["vforw_xy"] = vforw_xy * max_amp
    result["vrefl_xy"] = vrefl_xy * max_amp
    result["vforw_abcd"] = vforw_abcd * max_amp
    result["vrefl_abcd"] = vrefl_abcd * max_amp
    result["bandwidth_xy"] = bandwidth_xy
    result["detuning_xy"] = detuning_xy
    result["bandwidth_abcd"] = bandwidth_abcd
    result["detuning_abcd"] = detuning_abcd

    return result


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    F0 = 1.3e9
    DELAY_DECAY = 50
    FILLING_DELAY = 100
    PROBE_AMP_ADDR = "XFEL.RF/LLRF.CONTROLLER/C1.M1.A11.L3/PROBE.AMPL"
    PROBE_PHA_ADDR = "XFEL.RF/LLRF.CONTROLLER/C1.M1.A11.L3/PROBE.PHASE"
    FORWARD_AMP_ADDR = "XFEL.RF/LLRF.CONTROLLER/C1.M1.A11.L3/VFORW.AMPL"
    FORWARD_PHA_ADDR = "XFEL.RF/LLRF.CONTROLLER/C1.M1.A11.L3/VFORW.PHASE"
    REFLECTED_AMP_ADDR = "XFEL.RF/LLRF.CONTROLLER/C1.M1.A11.L3/VREFL.AMPL"
    REFLECTED_PHA_ADDR = "XFEL.RF/LLRF.CONTROLLER/C1.M1.A11.L3/VREFL.PHASE"
    DELAY_ADDR = "XFEL.RF/LLRF.CONTROLLER/CTRL.A11.L3/PULSE_DELAY"
    FILLING_ADDR = "XFEL.RF/LLRF.CONTROLLER/CTRL.A11.L3/PULSE_FILLING"
    FLATTOP_ADDR = "XFEL.RF/LLRF.CONTROLLER/CTRL.A11.L3/PULSE_FLATTOP"
    THRESHOLD = 0.70
    FLATTEN_S = 50e-6

    result = None

    ATTEMPTS = 3

    attempt = ATTEMPTS

    while attempt != 0:
        try:
            result = calculate_abcd(F0, DELAY_DECAY, FILLING_DELAY, 
                                    PROBE_AMP_ADDR, PROBE_PHA_ADDR, 
                                    FORWARD_AMP_ADDR, FORWARD_PHA_ADDR, 
                                    REFLECTED_AMP_ADDR, REFLECTED_PHA_ADDR, 
                                    DELAY_ADDR, FILLING_ADDR, FLATTOP_ADDR, 
                                    THRESHOLD, FLATTEN_S)
        except:
            attempt -= 1

    if not result:
        print("Failed to calibrate the signal after", ATTEMPTS, "attempts")
    else:
        print("QL", result["QL"])
        print("xy   coefficients", result["xy"])
        print("abcd coefficients", result["abcd"])

        plt.figure()
        plt.plot(result["time_trace"], result["detuning_abcd"])
        
        plt.figure()
        plt.plot(result["time_trace"], result["bandwidth_abcd"])
        
        plt.figure()
        plt.plot(result["time_trace"], np.real(result["probe"]))
        plt.plot(result["time_trace"], np.real(result["vforw_abcd"] + result["vrefl_abcd"]))
        
        plt.figure()
        plt.plot(result["time_trace"], np.imag(result["probe"]))
        plt.plot(result["time_trace"], np.imag(result["vforw_abcd"] + result["vrefl_abcd"]))

        plt.show()
