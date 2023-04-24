import pydoocs
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
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
    start = delay + filling_delay
    stop =  delay + filling + min(flattop, decay_length)
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

def calibrate_abcd(time_trace, hbw,
                   probe_pulse_cmplx, forward_pulse_cmplx, reflected_pulse_cmplx,
                   probe_decay_cmplx, forward_decay_cmplx, reflected_decay_cmplx,
                   QL_weight = 1):

    (probe_pulse_re, probe_pulse_im)         = C2REIM(probe_pulse_cmplx)
    (forward_pulse_re, forward_pulse_im)     = C2REIM(forward_pulse_cmplx)
    (reflected_pulse_re, reflected_pulse_im) = C2REIM(reflected_pulse_cmplx)
    (probe_decay_re, probe_decay_im)         = C2REIM(probe_decay_cmplx)
    (forward_decay_re, forward_decay_im)     = C2REIM(forward_decay_cmplx)
    (reflected_decay_re, reflected_decay_im) = C2REIM(reflected_decay_cmplx)

    A2 = np.abs(probe_pulse_cmplx)**2
    dt = (time_trace[1] - time_trace[0]) * 1e-6

    zeros_pulse = np.zeros_like(probe_pulse_re)
    zeros_decay = np.zeros_like(probe_decay_re)

    A_pulse_re = [forward_pulse_re, -forward_pulse_im, 
                  reflected_pulse_re, -reflected_pulse_im] * 2

    A_pulse_im = [forward_pulse_im,  forward_pulse_re, 
                  reflected_pulse_im, -reflected_pulse_re] * 2 

    A_pulse_Q  = ([4 * QL_weight * (  probe_pulse_re * forward_pulse_re   + probe_pulse_im * forward_pulse_im), 
                   4 * QL_weight * (- probe_pulse_re * forward_pulse_im   + probe_pulse_im * forward_pulse_re),
                   4 * QL_weight * (  probe_pulse_re * reflected_pulse_re + probe_pulse_im * reflected_pulse_im),
                   4 * QL_weight * (- probe_pulse_re * reflected_pulse_im + probe_pulse_im * reflected_pulse_re)] + 
                  [zeros_pulse] * 4)

    A_decay_vforw_re = [forward_decay_re, -forward_decay_im, 
                        reflected_decay_re, -reflected_decay_im] + [zeros_decay] * 4

    A_decay_vforw_im = [forward_decay_im,  forward_decay_re, 
                        reflected_decay_im, -reflected_decay_re] + [zeros_decay] * 4

    A_decay_vrefl_re = ([zeros_decay] * 4) + [forward_decay_re, -forward_decay_im, 
                                              reflected_decay_re, -reflected_decay_im]

    A_decay_vrefl_im = ([zeros_decay] * 4) + [forward_decay_im,  forward_decay_re, 
                                              reflected_decay_im, -reflected_decay_re]

    A_pulse_re       = np.column_stack(A_pulse_re)
    A_pulse_im       = np.column_stack(A_pulse_im)
    A_pulse_Q        = np.column_stack(A_pulse_Q)
    A_decay_vforw_re = np.column_stack(A_decay_vforw_re)
    A_decay_vforw_im = np.column_stack(A_decay_vforw_im)
    A_decay_vrefl_re = np.column_stack(A_decay_vrefl_re)
    A_decay_vrefl_im = np.column_stack(A_decay_vrefl_im)   

    A = np.vstack((A_pulse_re, 
                   A_pulse_im,     
                   A_pulse_Q,       
                   A_decay_vforw_re,
                   A_decay_vforw_im, 
                   A_decay_vrefl_re, 
                   A_decay_vrefl_im))

    b_pulse_re       = probe_pulse_re
    b_pulse_im       = probe_pulse_im

    A2_deriv = savgol_filter(A2, 101, 3, deriv=1, delta=dt)

    b_pulse_Q        = QL_weight * (A2_deriv/ (2 * np.pi * hbw) + 2 * A2)
    b_decay_vforw_re = zeros_decay
    b_decay_vforw_im = zeros_decay
    b_decay_vrefl_re = probe_decay_re
    b_decay_vrefl_im = probe_decay_im

    b = np.concatenate((b_pulse_re, 
                        b_pulse_im, 
                        b_pulse_Q, 
                        b_decay_vforw_re, 
                        b_decay_vforw_im, 
                        b_decay_vrefl_re, 
                        b_decay_vrefl_im))

    b = np.reshape(b, (b.shape[0], 1))

    x = np.linalg.lstsq(A, b, rcond=None)[0]

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

    QL = find_QL(f0, probe_amp_addr, threshold, delay_decay, delay_addr, filling_addr, flattop_addr)
    hbw = 0.5 * f0 / QL
    decay_length = np.pi * QL / f0 * 1e6
    (time_trace, probe, vforw_orig, vrefl_orig) = get_traces_cmplx(probe_amp_addr, probe_pha_addr,
                                                                   vforw_amp_addr, vforw_pha_addr,
                                                                   vrefl_amp_addr, vrefl_pha_addr)
    dt = (time_trace[1] - time_trace[0]) * 1e-6

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

    def calibrate_abcd_fun(hbw_t):
        abcd = calibrate_abcd(time_trace, hbw_t,
                              probe[start_pulse_idx:stop_pulse_idx],
                              vforw_xy[start_pulse_idx:stop_pulse_idx],
                              vrefl_xy[start_pulse_idx:stop_pulse_idx],
                              probe[start_decay_idx:stop_decay_idx],
                              vforw_xy[start_decay_idx:stop_decay_idx],
                              vrefl_xy[start_decay_idx:stop_decay_idx])

        return abcd

    def get_abcd_loss(hbw_t):
        abcd = calibrate_abcd_fun(hbw_t)
        a = abcd[0][0]
        b = abcd[1][0]
        c = abcd[2][0]
        d = abcd[3][0]

        vforw_abcd = a * vforw_xy + b * vrefl_xy 
        vrefl_abcd = c * vforw_xy + d * vrefl_xy

        (bandwidth_abcd, _) = compute_bwdet(time_trace, hbw, probe, vforw_abcd, vrefl_abcd)

        bandwidth_abcd[np.isnan(bandwidth_abcd)] = 0.0

        if int(flatten_s/dt) * 2 + 1 > 3:
            bandwidth_abcd  = savgol_filter(bandwidth_abcd,  int(flatten_s/dt) * 2 + 1, 3)

        return np.var(bandwidth_abcd[start_idx:-start_idx])


    hbws = np.linspace(1-FRAC_OPT, 1+FRAC_OPT, ITER_OPT) * hbw
    var = []

    for i, hbw_t in enumerate(hbws):
        var.append(get_abcd_loss(hbw_t))

    hbw_new = hbws[np.argmin(var)]
    hbw = hbw_new
    QL = 0.5 * f0 / hbw

    abcd = calibrate_abcd_fun(hbw)

    a = abcd[0][0]
    b = abcd[1][0]
    c = abcd[2][0]
    d = abcd[3][0]

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
