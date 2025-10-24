from psp.dsp.filters import true_rms
import comtrade
import numpy as np


# Associated with testfile1_50Hz.cfg
def test_rms_1():
    filename_cfg = "./tests/data/50Hz/testfile1_50Hz.CFG"
    filename_dat = "./tests/data/50Hz/testfile1_50Hz.DAT"
    rec = comtrade.load(filename_cfg, filename_dat)

    freq = rec.cfg.frequency  # system frequency
    srate = rec.cfg.sample_rates[0][0]  # sampling rate
    N = int(srate / freq)  # Number of sample per cycle

    UL1 = np.array(rec.analog[0])
    IL1 = np.array(rec.analog[1])

    UL1_rms = true_rms(UL1, N)
    IL1_rms = true_rms(IL1, N)

    UL1_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile1_50Hz_UL1_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile1_50Hz_IL1_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )

    # np.allclose(A,B,...)
    assert np.array_equal(UL1_rms, UL1_rms_res)
    assert np.array_equal(IL1_rms, IL1_rms_res)


# Associated with testfile2_50Hz.cfg
def test_rms_2():
    filename_cfg = "./tests/data/50Hz/testfile2_50Hz.CFG"
    filename_dat = "./tests/data/50Hz/testfile2_50Hz.DAT"
    rec = comtrade.load(filename_cfg, filename_dat)

    freq = rec.cfg.frequency  # system frequency
    srate = rec.cfg.sample_rates[0][0]  # sampling rate
    N = int(srate / freq)  # Number of sample per cycle

    UL1 = np.array(rec.analog[0])
    UL2 = np.array(rec.analog[1])
    UL3 = np.array(rec.analog[2])
    IL1 = np.array(rec.analog[3])
    IL2 = np.array(rec.analog[4])
    IL3 = np.array(rec.analog[5])

    UL1_rms = true_rms(UL1, N)
    UL2_rms = true_rms(UL2, N)
    UL3_rms = true_rms(UL3, N)
    IL1_rms = true_rms(IL1, N)
    IL2_rms = true_rms(IL2, N)
    IL3_rms = true_rms(IL3, N)

    UL1_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_UL1_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL2_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_UL2_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL3_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_UL3_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL1_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL2_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL2_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL3_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL3_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )

    # np.allclose(A,B,...)
    assert np.array_equal(abs(UL1_rms), UL1_rms_res)
    assert np.array_equal(abs(UL2_rms), UL2_rms_res)
    assert np.array_equal(abs(UL3_rms), UL3_rms_res)
    assert np.array_equal(abs(IL1_rms), IL1_rms_res)
    assert np.array_equal(abs(IL2_rms), IL2_rms_res)
    assert np.array_equal(abs(IL3_rms), IL3_rms_res)


# Associated with testfile3_50Hz.cfg
def test_rms_3():
    filename_cfg = "./tests/data/50Hz/testfile3_50Hz.CFG"
    filename_dat = "./tests/data/50Hz/testfile3_50Hz.DAT"
    rec = comtrade.load(filename_cfg, filename_dat)

    freq = rec.cfg.frequency  # system frequency
    srate = rec.cfg.sample_rates[0][0]  # sampling rate
    N = int(srate / freq)  # Number of sample per cycle

    UL1 = np.array(rec.analog[0])
    UL2 = np.array(rec.analog[1])
    UL3 = np.array(rec.analog[2])
    IL1 = np.array(rec.analog[3])
    IL2 = np.array(rec.analog[4])
    IL3 = np.array(rec.analog[5])

    UL1_rms = true_rms(UL1, N)
    UL2_rms = true_rms(UL2, N)
    UL3_rms = true_rms(UL3, N)
    IL1_rms = true_rms(IL1, N)
    IL2_rms = true_rms(IL2, N)
    IL3_rms = true_rms(IL3, N)

    UL1_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_UL1_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL2_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_UL2_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL3_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_UL3_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL1_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL2_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL2_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL3_rms_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL3_rms.csv",
        delimiter=",",
        dtype=np.float64,
    )

    # np.allclose(A,B,...)
    assert np.array_equal(abs(UL1_rms), UL1_rms_res)
    assert np.array_equal(abs(UL2_rms), UL2_rms_res)
    assert np.array_equal(abs(UL3_rms), UL3_rms_res)
    assert np.array_equal(abs(IL1_rms), IL1_rms_res)
    assert np.array_equal(abs(IL2_rms), IL2_rms_res)
    assert np.array_equal(abs(IL3_rms), IL3_rms_res)
