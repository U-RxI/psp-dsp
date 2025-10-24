# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:18:32 2024

@author: ASGRM
"""

from scipy.fft import fft
import numpy as np
from numpy.typing import NDArray
from typing import Iterable


def dft(sample: Iterable[float], N: int, harmonic: int = 1) -> NDArray[complex]:
    """
    A function to filter a signal using discrete fourier transformation.

    The function it intended for analyzing a measured power system voltage
    or current.
    The filtering is done with a moving window of N samples, where only the
    specified harmonic component is kept. Each calculated point is a complex
    number x+1jy, where the magnitude is the RMS and the argument is the angle:
    abs(x+1jy) = RMS / math.atan2(y, x) => angle.
    The initial windows is initialized with zeros.
    This yields len(sample) == len(result).

    Parameters
    ----------
    sample : Iterable[float]
        Array with data to be filtered.
    N : int
        Number of samples for the dft window.
        For a one-cycle filter at 50Hz with a sampling rate of 1000Hz:
        N = int(50Hz/1000Hz) = 20
    harmonic : int, optional
        Harmonic component to be filtered, where 1 is the fundamental.
        The default is 1.

    Returns
    -------
    result : NDArray[complex]
        Numpy array with the result.

    """

    result = np.empty(0)  # Zero padding

    window = np.zeros(N)  # Init of the window

    for value in sample:
        window = np.append(window[1:N], value)

        yff = fft(window)

        component = yff[harmonic]

        component_rms = 2 / N * component / np.sqrt(2)

        result = np.append(result, component_rms)

    return result


def true_rms(sample: Iterable[float], N: int):
    true_rms = np.empty(0)
    window = np.zeros(N)  # Init window

    for value in sample:
        window = np.append(window[1:N], value)
        true_rms = np.append(true_rms, np.sqrt(1 / N * np.sum(window**2)))

    return true_rms


def reconstruct_signal(dft_output, N, harmonic=1):
    """
    Reconstruct the original signal from the DFT output.

    Parameters:
        dft_output (array-like): The array of complex numbers (DFT output).
        N (int): Number of samples in one period of the fundamental.
        harmonic (int): The harmonic being used for reconstruction.

    Returns:
        np.ndarray: The reconstructed signal.
    """
    # Time vector for one period
    t = np.arange(N) / N  # Normalized time vector (0 to 1)

    # Initialize the reconstructed signal
    reconstructed = np.zeros(len(dft_output) * N)

    for i, component in enumerate(dft_output):
        # Calculate the real and imaginary contributions for this harmonic
        amplitude = np.abs(component)  # Magnitude of the component
        phase = np.angle(component)  # Phase angle of the component

        # Reconstruct the signal for one period
        period_signal = (
            2 ** (1 / 2) * amplitude * np.cos(2 * np.pi * harmonic * t + phase)
        )

        # Append the period signal to the full reconstructed signal
        reconstructed[i * N : (i + 1) * N] = period_signal

    return reconstructed


"""

#To do
#Bench mark wavewin/sigra
#Commments and descriptions
#Half cycle
#Cosine
#padding?
#
# 3ms / 8ms
# Time selector?

# Testing
# Draw fundamental waveform

import matplotlib.pyplot as plt
import comtrade

rec = comtrade.load("test_file/Case1/AA3F1Q13FN1_DR7_20220417083945.cfg", "test_file/Case1/AA3F1Q13FN1_DR7_20220417083945.dat", use_double_precision=False)



N = 20

t = rec.time
IA = rec.analog[0]
IB = rec.analog[1]
IC = rec.analog[2]


IA_dft = dft(IA, N)
IB_dft = dft(IB, N)
IC_dft = dft(IC, N)

 

IA_true_rms = true_rms(IA, N)
IB_true_rms = true_rms(IB, N)
IC_true_rms = true_rms(IC, N)


plt.figure()
#plt.plot(rec.time, rec.analog[2])
plt.plot(rec.time, reconstruct_signal(IC_dft, N)[0::N] )
#plt.plot(rec.time, reconstruct_signal(IC_dft, N)[5::N] )
#plt.plot(rec.time, reconstruct_signal(IC_dft, N)[10::N] )
#plt.plot(rec.time, reconstruct_signal(IC_dft, N)[15::N] )
#plt.plot(rec.time, reconstruct_signal(IC_dft, N)[20::N] )

plt.legend([rec.analog_channel_ids[0], rec.analog_channel_ids[1], rec.analog_channel_ids[2]])
plt.show()



plt.figure()
plt.title('Fundamental rms')
plt.plot(t, abs(IC_dft))

t_rms = true_rms(IC, N)

plt.figure()
plt.title('True rms')
plt.plot(t, t_rms)




N = 20


from numpy import genfromtxt
#IA_fund_2 = genfromtxt('test_file/Case1/out.csv', usecols = (2), skip_header=1, delimiter=',')
#IB_fund_2 = genfromtxt('test_file/Case1/out.csv', usecols = (3), skip_header=1, delimiter=',')
#IC_fund_2 = genfromtxt('test_file/Case1/out.csv', usecols = (4), skip_header=1, delimiter=',')
IA_trms_2 = genfromtxt('test_file/Case1/out_true_rms.csv', usecols = (2), skip_header=1, delimiter=',')
IB_trms_2 = genfromtxt('test_file/Case1/out_true_rms.csv', usecols = (3), skip_header=1, delimiter=',')
IC_trms_2 = genfromtxt('test_file/Case1/out_true_rms.csv', usecols = (4), skip_header=1, delimiter=',')





rec2 = comtrade.load("test_file/Case1/rms.cfg", "test_file/Case1/rms.dat")

IA2 = rec.analog[0]
IB2 = rec.analog[1]
IC2 = rec.analog[2]

IA2_dft_mag = rec.analog[3]


plt.figure()
plt.title('From wavewin')
plt.plot(rec2.time, abs(dft(IA2_dft_mag,N)))







plt.show()


# Sigra
# a = abs(-499.6-808.4) / 1309
a = abs(-499.6-808.4) / 1309

from numpy import angle, cos, sin

def intp(x, array):
    # Interpolation between two data points
    idx = int(x)
    x1, y1 = (idx, abs(array[idx]))
    x2, y2 = (idx+1, abs(array[idx+1]))
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    mag = a*x + b
    
    x1, y1 = (idx, angle(array[idx]))
    x2, y2 = (idx+1, angle(array[idx+1]))
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    
    
    ang = a*x + b
    
    
    return mag*sin(ang) + mag*cos(ang)*1j

# polar_str(intp(rec.trigger_time*1e3, IC_dft) / P2R(1, angle(intp(rec.trigger_time*1e3, IA_dft)) / pi * 180))


# ---------------------------
# Integrety check

if not rec._use_double_precision:

    # first 20 samples (0-19ms)
    ## Phase A - first 20 samples (0-19ms)
    assert round(IA_dft[N-1].real, 8) == 105.25688606, 'Error cal dft phase A'
    assert round(IA_dft[N-1].imag, 8) == -41.49883936, 'Error cal dft phase A'
    
    ## Phase B - first 20 samples (0-19ms)
    assert round(IB_dft[N-1].real, 8) == -88.15303484, 'Error cal dft phase B'
    assert round(IB_dft[N-1].imag, 8) == -70.83101706, 'Error cal dft phase B'
    
    ## Phase C - first 20 samples (0-19ms)
    assert round(IC_dft[N-1].real, 8) == -16.42404135, 'Error cal dft phase C'
    assert round(IC_dft[N-1].imag, 8) == 111.40946317, 'Error cal dft phase C'
    
    # at trigger point sample 500
    ## Phase A
    assert round(IA_dft[500].real, 8) == 2.5975461, 'Error cal dft phase A'
    assert round(IA_dft[500].imag, 8) == -22.77354464, 'Error cal dft phase A'

    ## Phase B
    assert round(IB_dft[500].real, 8) == -52.61922361, 'Error cal dft phase B'
    assert round(IB_dft[500].imag, 8) == -107.98435017, 'Error cal dft phase B'

    ## Phase C
    assert round(IC_dft[500].real, 8) == 805.83971603, 'Error cal dft phase C'
    assert round(IC_dft[500].imag, 8) == -35.20313751, 'Error cal dft phase C'

if rec._use_double_precision:
    ## Phase A - first 20 samples (0-19ms)
    assert round(IA_dft[N-1].real, 8) == 105.25688506, 'Error cal dft phase A'
    assert round(IA_dft[N-1].imag, 8) == -41.49883849, 'Error cal dft phase A'
    
    ## Phase B - first 20 samples (0-19ms)
    assert round(IB_dft[N-1].real, 8) == -88.1530356, 'Error cal dft phase B'
    assert round(IB_dft[N-1].imag, 8) == -70.83101759, 'Error cal dft phase B'
    
    ## Phase C - first 20 samples (0-19ms)
    assert round(IC_dft[N-1].real, 8) == -16.42404094, 'Error cal dft phase C'
    assert round(IC_dft[N-1].imag, 8) == 111.40946433, 'Error cal dft phase C'
    
    # at trigger point sample 500
    ## Phase A
    assert round(IA_dft[500].real, 8) == 2.5975466, 'Error cal dft phase A'
    assert round(IA_dft[500].imag, 8) == -22.77354462, 'Error cal dft phase A'

    ## Phase B
    assert round(IB_dft[500].real, 8) == -52.61922477, 'Error cal dft phase B'
    assert round(IB_dft[500].imag, 8) == -107.98435149, 'Error cal dft phase B'

    ## Phase C
    assert round(IC_dft[500].real, 8) == 805.83971808, 'Error cal dft phase C'
    assert round(IC_dft[500].imag, 8) == -35.20313703, 'Error cal dft phase C'

# ---------------------------



## Compare with Wavewin (v H.P.9)
### fundamental RMS
assert round(abs(IA_dft[N-1]), 3) == 113.142, 'Error cal dft rms phase A'
assert round(abs(IB_dft[N-1]), 3) == 113.084, 'Error cal dft rms phase B'
assert round(abs(IC_dft[N-1]), 3) == 112.614, 'Error cal dft rms phase C'
### true RMS
assert round(abs(IA_true_rms[N-1]), 3) == 113.158, 'Error cal dft true rms phase A'
assert round(abs(IB_true_rms[N-1]), 3) == 113.099, 'Error cal dft true rms phase B'
assert round(abs(IC_true_rms[N-1]), 3) == 112.629, 'Error cal dft true rms phase C'

# ---------------------------

# ---------------------------
# at trigger time (0.5 s)

## since sampling rate is 20 samples per cycle index to the trigger time is 
## calculated by multipling with 1e3 (ms)
x = round(rec.trigger_time*1e3) # (0.5 s)

assert round(IA_dft[x].real, 8) == -7.12473429, 'Error cal dft phase A'
assert round(IA_dft[x].imag, 8) == -22.46161392, 'Error cal dft phase A'

assert round(IB_dft[x].real, 8) == -82.59266447, 'Error cal dft phase B'
assert round(IB_dft[x].imag, 8) == -86.43898556, 'Error cal dft phase B'

assert round(IC_dft[x].real, 8) == 668.93075739, 'Error cal dft phase C'
assert round(IC_dft[x].imag, 8) == -282.49834032, 'Error cal dft phase C'

## Compare with Wavewin (v H.P.9)
### fundamental RMS
assert round(abs(IA_dft[x]), 3) == 23.565, 'Error cal dft rms phase A'
assert round(abs(IB_dft[x]), 3) == 119.554, 'Error cal dft rms phase B'
assert round(abs(IC_dft[x]), 3) == 726.136, 'Error cal dft rms phase C'
### true RMS
assert round(abs(IA_true_rms[x]), 3) == 166.081, 'Error cal dft true rms phase A'
assert round(abs(IB_true_rms[x]), 3) == 151.210, 'Error cal dft true rms phase B'
assert round(abs(IC_true_rms[x]), 3) == 967.478, 'Error cal dft true rms phase C'


"""
