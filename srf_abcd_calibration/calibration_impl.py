# Copyright 2023 A. Bellandi et al.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
    Calibration methods for SRF cavity accelerating systems. See:
    Bellandi, Andrea, et al. 'Calibration of superconducting
    radio-frequency cavity forward and reflected channels based on stored energy dynamics'
"""

# Params
#
# hbw: Cavity half bandwidth in angular frequency
# probe_cmplx, vforw_cmplx, vrefl_cmplx: Cavity signal traces in I (real) and Q (imaginary)
# probe_cmplx_decay, vforw_cmplx_decay, vrefl_cmplx_decay: Cavity signals on decay
# probe_sq_deriv: time derivative of the probe square amplitude
# kadd: tuning parameter
#
# The calibration algorithms returns a 4 complex values array with
#
# (a, b, c, d) = (arr[0], arr[1], arr[2], arr[3])

import numpy as np
from scipy.optimize import least_squares, lsq_linear

# -------------------------- Utility functions --------------------------------

def C2RE(x):
    """
        Separate the real (even indices) from imaginary (odd indices) parts
        of a complex array in a real array
    """
    result = np.empty(2*np.array(x).shape[0], dtype=float)
    result[0::2] = np.real(x)
    result[1::2] = np.imag(x)
    return result


def RE2C(x):
    """
        Merge the real (even indices) and imaginary (odd indices) parts
        of a real array in a complex array
    """
    x = np.array(x)
    return x[0::2] + 1.0j * x[1::2]


# -------------------------- Calibration methods ------------------------------

def calibrate_diagonal(probe_cmplx, vforw_cmplx, vrefl_cmplx):
    """
        Classical calibration method. b,c terms are assumed to be zero and
        probe = a*vforw + d*vrefl
    """
    A = np.empty((probe_cmplx.shape[0], 2), dtype=complex)
    A[:, 0] = vforw_cmplx
    A[:, 1] = vrefl_cmplx

    b = probe_cmplx
    calib = lsq_linear(A, b).x

    return np.array([calib[0], 0, 0, calib[1]])


def calibrate_from_ref_7_kadd_1(probe_cmplx, vforw_cmplx, vrefl_cmplx,
        probe_cmplx_decay, vforw_cmplx_decay, vrefl_cmplx_decay, kadd=1):
    """
        Method from:
        Pfeiffer, Sven, et al. "Virtual cavity probe generation using calibrated
        forward and reflected signals." MOPWA040, IPAC, 2015, 15.
        The parameter 'kadd' is defaulted to 1
    """

    zeros = np.zeros_like(vforw_cmplx_decay)

    A_probe = np.column_stack([vforw_cmplx, vrefl_cmplx] * 2)
    A_vforw_cmplx_decay = np.column_stack([vforw_cmplx_decay, vrefl_cmplx_decay, zeros, zeros])
    A_vrefl_cmplx_decay = np.column_stack([zeros, zeros, vforw_cmplx_decay, vrefl_cmplx_decay])

    (x, _, _, y) = tuple(calibrate_diagonal(probe_cmplx, vforw_cmplx, vrefl_cmplx))

    S = lsq_linear(np.column_stack([-vrefl_cmplx_decay]), vforw_cmplx_decay).x[0]

    Wb = np.abs(S)
    Wc = kadd*Wb

    A_absx = np.column_stack([[np.abs(x)-Wc], [0.0], [1.0/Wc], [0.0]])
    A_absy = np.column_stack([[0.0], [1.0/Wb], [0.0], np.abs(y)-Wb])

    A = np.vstack([A_probe, A_vforw_cmplx_decay, A_vrefl_cmplx_decay, A_absx, A_absy])
    b = np.concatenate([probe_cmplx, zeros, probe_cmplx_decay, [np.abs(x)], [np.abs(y)]])

    return lsq_linear(A, b).x


def calibrate_energy(hbw, probe_cmplx, vforw_cmplx, vrefl_cmplx, probe_sq_deriv,
        probe_cmplx_decay=None, vforw_cmplx_decay=None, vrefl_cmplx_decay=None):
    """
        Cavity stored energy-based calibration method.
        If the decay traces are assigned, the algorithm imposes a zero forward
        in the decay phase.
    """

    max_probe_recip = 1.0/np.max(np.abs(probe_cmplx))
    probe_cmplx_conj = np.conjugate(probe_cmplx)
    C = probe_sq_deriv/(2*hbw)
    D = C + np.abs(probe_cmplx)**2

    if (probe_cmplx_decay is None or vforw_cmplx_decay is None or vrefl_cmplx_decay is None):
        probe_cmplx_decay = np.zeros(0)
        vforw_cmplx_decay = np.zeros(0)
        vrefl_cmplx_decay = np.zeros(0)

    # Optimization routine. least squares tries to minimize  ||fun(abcd)||
    def fun(abcd):
        abcd = RE2C(abcd)
        vforw_calib = abcd[0] * vforw_cmplx + abcd[1] * vrefl_cmplx
        vrefl_calib = abcd[2] * vforw_cmplx + abcd[3] * vrefl_cmplx

        vforw_calib_decay = abcd[0] * vforw_cmplx_decay + abcd[1] * vrefl_cmplx_decay
        vrefl_calib_decay = abcd[2] * vforw_cmplx_decay + abcd[3] * vrefl_cmplx_decay

        # Error of Eq. 5
        dprobe = vforw_calib + vrefl_calib - probe_cmplx

        # Error of Eq. 10
        dD = (2.0 * np.real(probe_cmplx_conj * vforw_calib) - D) * max_probe_recip

        # Error of Eq. 11
        dC = (np.abs(vforw_calib)**2 - np.abs(vrefl_calib)**2 - C) * max_probe_recip

        # Error of Eq. 8
        dvforw_calib_decay = vforw_calib_decay

        return C2RE(np.concatenate([dprobe, dD, dC, dvforw_calib_decay]))

        # The initial guess for the least squares algorithm is (a=1, b=0, c=0, d=1)
    return RE2C(least_squares(fun, C2RE([1.0, 0.0, 0.0, 1.0]), method="lm").x)

