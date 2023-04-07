import tkinter as tk
import pydoocs
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from time import sleep

import matplotlib.pyplot as plt

F0 = 1.3e9
DELAY = 1
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
AVG = 1

fig, ((ax_I, ax_Q), (ax_det, ax_bw)) = plt.subplots(2, 2, sharex=True)

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

def get_regions_duration():
    return (pydoocs.read(DELAY_ADDR)["data"],
            pydoocs.read(FILLING_ADDR)["data"],
            pydoocs.read(FLATTOP_ADDR)["data"])

def get_range_idx(time_trace, decay_length):
    (delay, filling, flattop) = get_regions_duration()
    start = delay + FILLING_DELAY
    stop =  delay + filling + flattop + decay_length
    return (find_nearest(time_trace, start), find_nearest(time_trace, stop))

def get_pulse_range_idx(time_trace, decay_length):
    (delay, filling, flattop) = get_regions_duration()
    start = delay + FILLING_DELAY
    stop =  delay + filling + min(flattop, decay_length)
    return (find_nearest(time_trace, start), find_nearest(time_trace, stop))

def get_decay_range_idx(time_trace, decay_length):
    (delay, filling, flattop) = get_regions_duration()
    start = delay + filling + flattop + DELAY_DECAY
    stop = delay + filling + flattop + decay_length
    return (find_nearest(time_trace, start), find_nearest(time_trace, stop))

def find_QL():

    try:
        probe = pydoocs.read(PROBE_AMP_ADDR)["data"]
        (delay, filling, flattop) = get_regions_duration()

        def exponential_decay(t, a, b):
            return a * np.exp(-t * np.pi * F0 / b)

        time_trace = probe[:, 0]
        amplitude = probe[:, 1]

        decay_start_idx = get_decay_range_idx(time_trace, 0)[0]

        time_trace = time_trace[decay_start_idx:]
        time_trace = time_trace - time_trace[0]
        amplitude = amplitude[decay_start_idx:]

        decay_stop_idx = find_nearest(amplitude, amplitude[0] * THRESHOLD)
        time_trace = time_trace[:decay_stop_idx]
        amplitude = amplitude[:decay_stop_idx]

        popt, pcov = curve_fit(exponential_decay, 
                               time_trace / 1e6, amplitude,
                               p0 = [amplitude[0], 1e7])

        return popt[1]
    except:
        return None

def get_traces_cmplx(avg=1):

    time_trace = None

    probe_cmplx = []
    forward_cmplx = []
    reflected_cmplx = []

    for _ in range(avg):
        probe_amp = pydoocs.read(PROBE_AMP_ADDR)
        macropulse = probe_amp["macropulse"]
        time_trace = probe_amp["data"][:, 0]
        probe_amp = probe_amp["data"][:, 1]
        probe_pha = pydoocs.read(PROBE_PHA_ADDR, macropulse=macropulse)["data"][:, 1]
        forward_amp = pydoocs.read(FORWARD_AMP_ADDR, macropulse=macropulse)["data"][:, 1]
        forward_pha = pydoocs.read(FORWARD_PHA_ADDR, macropulse=macropulse)["data"][:, 1]
        reflected_amp = pydoocs.read(REFLECTED_AMP_ADDR, macropulse=macropulse)["data"][:, 1]
        reflected_pha = pydoocs.read(REFLECTED_PHA_ADDR, macropulse=macropulse)["data"][:, 1]

        probe_cmplx.append(AP2C(probe_amp, probe_pha))
        forward_cmplx.append(AP2C(forward_amp, forward_pha))
        reflected_cmplx.append(AP2C(reflected_amp, reflected_pha))

        sleep(DELAY)

    return (time_trace, 
            np.mean(probe_cmplx, axis=0),
            np.mean(forward_cmplx, axis=0),
            np.mean(reflected_cmplx, axis=0))


def calibrate_xy(probe_cmplx, forward_cmplx, reflected_cmplx):
    cols = probe_cmplx.shape[0]
    A = np.zeros((cols, 2), dtype=complex)
    A[:, 0] = forward_cmplx
    A[:, 1] = reflected_cmplx

    return np.linalg.lstsq(A, probe_cmplx, rcond=None)[0]

def calibrate_abcd(time_trace, hbw,
                   probe_pulse_cmplx, forward_pulse_cmplx, reflected_pulse_cmplx,
                   probe_decay_cmplx, forward_decay_cmplx, reflected_decay_cmplx):

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

    A_pulse_Q  = ([4 * (  probe_pulse_re * forward_pulse_re   + probe_pulse_im * forward_pulse_im), 
                   4 * (- probe_pulse_re * forward_pulse_im   + probe_pulse_im * forward_pulse_re),
                   4 * (  probe_pulse_re * reflected_pulse_re + probe_pulse_im * reflected_pulse_im),
                   4 * (- probe_pulse_re * reflected_pulse_im + probe_pulse_im * reflected_pulse_re)] + 
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

    A2_deriv = np.gradient(A2) / dt

    b_pulse_Q        = np.gradient(A2) / (2 * np.pi * hbw * dt) + 2 * A2
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
    bwdet = probe * np.conj(rt) / A2
    return (np.real(bwdet), -np.imag(bwdet))


while True:
    QL = find_QL()
    hbw = 0.5 * F0 / QL
    decay_length = np.pi * QL / F0 * 1e6
    (time_trace, probe_cmplx, forward_cmplx, reflected_cmplx) = get_traces_cmplx(AVG)
    dt = (time_trace[1] - time_trace[0]) * 1e-6

    max_amp = np.max(np.abs(probe_cmplx))
    
    probe_cmplx     /= max_amp
    forward_cmplx   /= max_amp
    reflected_cmplx /= max_amp

    (start_idx, stop_idx) = get_range_idx(time_trace, decay_length)
    
    xy = calibrate_xy(probe_cmplx[start_idx:stop_idx],
                      forward_cmplx[start_idx:stop_idx],
                      reflected_cmplx[start_idx:stop_idx])

    x = xy[0]
    y = xy[1]

    forward_cmplx *= x
    reflected_cmplx *= y

    (start_pulse_idx, stop_pulse_idx) = get_pulse_range_idx(time_trace, decay_length)
    (start_decay_idx, stop_decay_idx) = get_decay_range_idx(time_trace, decay_length)

    abcd = calibrate_abcd(time_trace, hbw,
                          probe_cmplx[start_pulse_idx:stop_pulse_idx],
                          forward_cmplx[start_pulse_idx:stop_pulse_idx],
                          reflected_cmplx[start_pulse_idx:stop_pulse_idx],
                          probe_cmplx[start_decay_idx:stop_decay_idx],
                          forward_cmplx[start_decay_idx:stop_decay_idx],
                          reflected_cmplx[start_decay_idx:stop_decay_idx])

    a = abcd[0]
    b = abcd[1]
    c = abcd[2]
    d = abcd[3]

    corr_forward   = a * forward_cmplx + b * reflected_cmplx
    corr_reflected = c * forward_cmplx + d * reflected_cmplx

    vprobe_cmplx = corr_forward + corr_reflected


    (bw, det) = compute_bwdet(time_trace, hbw, probe_cmplx, corr_forward, corr_reflected)

    dt = (time_trace[1] - time_trace[0]) * 1e-6

    bw  = savgol_filter(bw,  int(FLATTEN_S/dt) * 2 + 1, 3)
    det = savgol_filter(det, int(FLATTEN_S/dt) * 2 + 1, 3)

    ax_I.cla()
    ax_Q.cla()
    ax_det.cla()
    ax_bw.cla()

    ax_I.plot(time_trace, np.real(probe_cmplx))
    ax_I.plot(time_trace, np.real(vprobe_cmplx))

    ax_Q.plot(time_trace, np.imag(probe_cmplx))
    ax_Q.plot(time_trace, np.imag(vprobe_cmplx))


    after_delay = time_trace > 100

    ax_bw.plot(time_trace[after_delay], bw[after_delay])
    ax_det.plot(time_trace[after_delay], det[after_delay])

    fig.show()

    input("Press a key")

    print("\n")
    print("QL:", QL, "decay length (us):", decay_length)
    print("")
    print("x (a.u.):", np.abs(x), "x (deg):", np.angle(x, deg=True))
    print("y (a.u.):", np.abs(y), "y (deg):", np.angle(y, deg=True))
    print("")
    print("a:", a, "b:", b, "c:", c, "d:", d)
    print("a+c:", a+c, "b+d:", b+d)

