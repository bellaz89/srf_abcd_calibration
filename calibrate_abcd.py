import pydoocs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar, least_squares, lsq_linear
from scipy.signal import savgol_filter, freqz 
from scipy.linalg import inv
from time import sleep
import warnings

#from sysidentpy.model_structure_selection import FROLS
#from sysidentpy.basis_function._basis_function import Polynomial
#from sysidentpy.metrics import root_relative_squared_error
##from sysidentpy.utils.generate_data import get_siso_data
#from sysidentpy.utils.display_results import results
#from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
#from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation




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

def get_flattop_range_idx(time_trace, decay_length, filling_delay, delay_addr, filling_addr, flattop_addr):
    (delay, filling, flattop) = get_regions_duration(delay_addr, filling_addr, flattop_addr)
    start = delay + filling
    stop =  delay + filling + flattop
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
    print("start", time_trace[0], "stop", time_trace[1])
    amplitude = probe[:, 1]

    decay_start_idx = get_decay_range_idx(time_trace, 0, delay_decay, delay_addr, filling_addr, flattop_addr)[0]

    time_trace = time_trace[decay_start_idx:]
    time_trace = time_trace - time_trace[0]
    amplitude = amplitude[decay_start_idx:]
    print(amplitude)

    decay_stop_idx = find_nearest(amplitude, amplitude[0] * threshold)
    time_trace = time_trace[:decay_stop_idx]
    amplitude = amplitude[:decay_stop_idx]

    print(amplitude)

    popt, pcov = curve_fit(exponential_decay, 
                           time_trace / 1e6, amplitude,
                           p0 = [amplitude[0], 1e7])

    return popt[1]

def find_klfd(time_trace, probe_cmplx, detuning):
    probe_amp_sq = np.abs(probe_cmplx)**2

    def klfd_square(t, det0, klfd, det_slope):
        return det0 + klfd * np.interp(t, time_trace, probe_amp_sq) + det_slope * t


    popt, pcov = curve_fit(klfd_square, time_trace, detuning, p0 = [0, -1, 0])

    print("Slope", popt[2]/(time_trace[1]-time_trace[0])*1e6, "Hz/s")

    return (popt[0], popt[1], popt[2]*1e6)

#def find_klfd_adv(time_trace, probe_cmplx, detuning, det0):
#    trace_len = len(time_trace)
#    probe_amp_sq = (np.abs(probe_cmplx)**2).reshape((trace_len, 1))
#    detuning = np.copy(detuning).reshape((trace_len, 1))
#
#    basis_function = Polynomial(degree=1)
#    
#    model = FROLS(
#        order_selection=True,
#        n_info_values=12,
#        extended_least_squares=False,
#        ylag=4, xlag=2,
#        info_criteria='aic',
#        estimator='least_squares',
#        basis_function=basis_function)
#
#    model.fit(X=probe_amp_sq, y=detuning)
#
#    yhat = model.predict(X=probe_amp_sq, y=detuning)
#
#    rrse = root_relative_squared_error(detuning, yhat)
#    print(rrse)
#
#    r = pd.DataFrame(
#        results(
#            model.final_model, model.theta, model.err,
#            model.n_terms, err_precision=8, dtype='sci'
#            ),
#        columns=['Regressors', 'Parameters', 'ERR'])
#
#    fs = 1.0/time_trace[1]-time_trace[0]
#
#    num = [model.theta[4][0]]

    #print(num)
    #print(den)
    #h, w = freqz(den, num, worN=2**14, fs=fs)
    #print(h)
    #print(w)

    #plt.figure()
    #plt.plot(w, np.abs(h))
    #plt.show()

    print(r)
    print(model.final_model)
    print(model.theta)

    den = [1] + list((-model.theta[:4]).reshape(4))

    w, h = freqz(den)
    plt.figure()
    plt.plot(w, np.abs(h))
    plt.show()
    

    return (time_trace, yhat.reshape(trace_len))


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

    def opt_fun(x):
        vforw_c_cmplx = forward_cmplx * (x[0] + 1.0j * x[1]) + reflected_cmplx * (x[2] + 1.0j * x[3])
        vrefl_c_cmplx = forward_cmplx * (x[4] + 1.0j * x[5]) + reflected_cmplx * (x[6] + 1.0j * x[7])

        diff_U = np.abs(vforw_c_cmplx)**2 - np.abs(vrefl_c_cmplx)**2
        return np.concatenate((np.dot(A, x[:8]) - b, 
                               2*U - np.dot(A_Q, x) + 2*diff_U,
                               dU / (2 * np.pi * hbw) - 2*diff_U))

    x = least_squares(opt_fun, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], method="lm").x

    return (x[0] + 1.0j * x[1], 
            x[2] + 1.0j * x[3], 
            x[4] + 1.0j * x[5], 
            x[6] + 1.0j * x[7])

def calibrate_abcd_linear(hbw,
                          probe_cmplx, forward_cmplx, reflected_cmplx, dU, U, 
                          probe_cmplx_decay, forward_cmplx_decay, reflected_cmplx_decay,
                          QL_weight = 1, fraction=1):

    (probe_re, probe_im)         = C2REIM(probe_cmplx)
    (forward_re, forward_im)     = C2REIM(forward_cmplx)
    (reflected_re, reflected_im) = C2REIM(reflected_cmplx)
    (probe_decay_re, probe_decay_im)         = C2REIM(probe_cmplx_decay*fraction)
    (forward_decay_re, forward_decay_im) = C2REIM(forward_cmplx_decay*fraction)
    (reflected_decay_re, reflected_decay_im) = C2REIM(reflected_cmplx_decay*fraction)

    zeros = np.zeros_like(probe_re)
    zeros_decay = np.zeros_like(forward_decay_re)

    A_re = [forward_re, -forward_im, 
            reflected_re, -reflected_im] * 2

    A_im = [forward_im,  forward_re, 
            reflected_im, -reflected_re] * 2 

    A_Q  = ([4 * QL_weight * (  probe_re * forward_re   + probe_im * forward_im), 
             4 * QL_weight * (- probe_re * forward_im   + probe_im * forward_re),
             4 * QL_weight * (  probe_re * reflected_re + probe_im * reflected_im),
             4 * QL_weight * (- probe_re * reflected_im + probe_im * reflected_re)] + 
            [zeros] * 4)

    A_forward_decay_re = [forward_decay_re, -forward_decay_im, 
                          reflected_decay_re, -reflected_decay_im] + [zeros_decay] * 4

    A_forward_decay_im = [forward_decay_im,  forward_decay_re, 
                          reflected_decay_im, -reflected_decay_re] + [zeros_decay] * 4 

    A_reflected_decay_re = [zeros_decay] * 4 + [forward_decay_re, -forward_decay_im, 
                                                reflected_decay_re, -reflected_decay_im] 

    A_reflected_decay_im = [zeros_decay] * 4 + [forward_decay_im,  forward_decay_re, 
                                                reflected_decay_im, -reflected_decay_re] 

    A_re = np.column_stack(A_re)
    A_im = np.column_stack(A_im)
    A_Q  = np.column_stack(A_Q)
    A_forward_decay_re  = np.column_stack(A_forward_decay_re)
    A_forward_decay_im  = np.column_stack(A_forward_decay_im)
    A_reflected_decay_re  = np.column_stack(A_reflected_decay_re)
    A_reflected_decay_im  = np.column_stack(A_reflected_decay_im)

    A = np.vstack((A_re, 
                   A_im,     
                   A_Q,
                   A_forward_decay_re,
                   A_forward_decay_im))


    b_re = probe_re
    b_im = probe_im
    b_Q  = QL_weight * (dU/ (2 * np.pi * hbw) + 2 * U)
    b = np.concatenate((b_re, 
                        b_im, 
                        b_Q,
                        zeros_decay,
                        zeros_decay))


    x = lsq_linear(A, b).x

    return (x[0] + 1.0j * x[1], 
            x[2] + 1.0j * x[3], 
            x[4] + 1.0j * x[5], 
            x[6] + 1.0j * x[7])


def calibrate_abcd_sven(probe_cmplx, forward_cmplx, reflected_cmplx,
                        probe_cmplx_decay, forward_cmplx_decay, reflected_cmplx_decay,
                        kadd=1):


    (x, y) = calibrate_xy(probe_cmplx, forward_cmplx, reflected_cmplx)

    #probe_cmplx           = probe_cmplx.transpose()
    #forward_cmplx         = forward_cmplx.transpose() 
    #reflected_cmplx       = reflected_cmplx.transpose()      
    #probe_cmplx_decay     = probe_cmplx_decay.transpose()
    #forward_cmplx_decay   = forward_cmplx_decay.transpose()  
    #reflected_cmplx_decay = reflected_cmplx_decay.transpose()


    # define A and B to have always same computation formula, see below
    A = -reflected_cmplx_decay
    B = forward_cmplx_decay

    S = A.transpose().dot(B)/(A.transpose().dot(A))

    # compute x (=a+c) and y (=b+d) for entire pulse
    A = np.block([forward_cmplx, reflected_cmplx])
    B = probe_cmplx

    #coeff1 = inv(A.transpose().dot(A)).dot(A.transpose().dot(B))

    #x = coeff1[0]
    #y = coeff1[1]

    # select signals for coupling computation
    a1 = forward_cmplx
    a2 = reflected_cmplx
    a3 = forward_cmplx_decay
    a4 = reflected_cmplx_decay

    b1 = probe_cmplx
    b2 = probe_cmplx_decay

    # define zeros for "big" matrix
    za = np.zeros(len(a3))

    # put all vectors together to calibration in-/output matrix
    Wb = np.abs(S)
    Wc = kadd*Wb

    print(Wc)
    print(Wb)

    # here A and B are given by Eqn. 10 in Paper
    # with B = A * x with x =[a b c d]'
    b = np.concatenate([b1, b2, za, [np.abs(x)], [np.abs(y)]])

    A = np.vstack([np.column_stack([a1, a2, a1, a2]),
                   np.column_stack([za, za, a3, a4]),
                   np.column_stack([a3, a4, za, za]),
                   np.column_stack([np.abs(x)-Wc, 0, 1/Wc, 0]),
                   np.column_stack([[0], 1/Wb, 0, np.abs(y)-Wb])])


    x = lsq_linear(A, b).x

    return (x[0], 
            x[1],
            x[2], 
            x[3],)

def compute_bwdet(time_trace, hbw, probe, forward, reflected):

    dt = (time_trace[1] - time_trace[0]) * 1e-6
    probe_deriv = np.gradient(probe) / dt

    A2 = np.abs(probe) ** 2

    rt = 2*hbw*forward - probe_deriv / (2.0 * np.pi)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bwdet = probe * np.conj(rt) / A2
        return (2 * savgol_filter(np.real(bwdet), 201, 2), -savgol_filter(np.imag(bwdet), 201, 2))


def calculate_abcd(f0, delay_decay, filling_delay, 
                   probe_amp_addr, probe_pha_addr, 
                   vforw_amp_addr, vforw_pha_addr, 
                   vrefl_amp_addr, vrefl_pha_addr, 
                   delay_addr, filling_addr, flattop_addr, 
                   threshold, flatten_s):

    FILTER_RANGE = 201

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

    (start_flattop_idx, stop_flattop_idx) = get_flattop_range_idx(time_trace, 
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
    start_flattop_idx += FILTER_RANGE
    stop_flattop_idx -= FILTER_RANGE 
    start_decay_idx += FILTER_RANGE
    stop_decay_idx -= FILTER_RANGE 

    probe_r = np.concatenate((probe[start_pulse_idx:stop_pulse_idx], 
                              probe[start_flattop_idx:stop_flattop_idx],
                              probe[start_decay_idx:stop_decay_idx]))

    vforw_r = np.concatenate((vforw_xy[start_pulse_idx:stop_pulse_idx], 
                              vforw_xy[start_flattop_idx:stop_flattop_idx],
                              vforw_xy[start_decay_idx:stop_decay_idx]))

    vrefl_r = np.concatenate((vrefl_xy[start_pulse_idx:stop_pulse_idx], 
                              vrefl_xy[start_flattop_idx:stop_flattop_idx],
                              vrefl_xy[start_decay_idx:stop_decay_idx]))

    U_r = np.concatenate((U[start_pulse_idx:stop_pulse_idx], 
                          U[start_flattop_idx:stop_flattop_idx],
                          U[start_decay_idx:stop_decay_idx]))

    dU_r = np.concatenate((dU[start_pulse_idx:stop_pulse_idx], 
                           dU[start_flattop_idx:stop_flattop_idx],
                           dU[start_decay_idx:stop_decay_idx]))

    vforw_d = vforw_xy[start_decay_idx:stop_decay_idx]
    vrefl_d = vrefl_xy[start_decay_idx:stop_decay_idx]
    probe_d = probe[start_decay_idx:stop_decay_idx]

    abcd = calibrate_abcd(hbw, probe_r, vforw_r, vrefl_r, dU_r, U_r, QL_weight = 1)
    #abcd_linear = calibrate_abcd_linear(hbw, probe_r, vforw_r, vrefl_r, dU_r, U_r, probe_d, vforw_d, vrefl_d)
    #abcd_sven   = calibrate_abcd_sven(probe_r, vforw_r, vrefl_r, probe_d, vforw_d, vrefl_d)
    
    #print("abcd", abcd)
    #print("abcd_linear", abcd_linear)
    #print("abcd_sven", abcd_sven)

    (a, b, c, d) = abcd
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

    (det0, klfd, det_slope) = find_klfd(time_trace[start_pulse_idx:],
                                        probe[start_pulse_idx:]*max_amp,
                                        detuning_abcd[start_pulse_idx:])

    #lfd_adv = find_klfd_adv(time_trace[start_pulse_idx:],
    #                        probe[start_pulse_idx:]*max_amp,
    #                        detuning_abcd[start_pulse_idx:], det0)

    print("det0", det0, "Hz. kldf", klfd, "Hz.")

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
    result["det0"] = det0
    result["klfd"] = klfd
    result["det_slope"] = det_slope
    #result["lfd_adv"] = lfd_adv

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
