from psp.dsp.filters import dft
import comtrade
import numpy as np


def load_test_case():
    pass


# Associated with testfile1_50Hz.cfg
def test_dft_1():
    filename_cfg = "./tests/data/50Hz/testfile1_50Hz.CFG"
    filename_dat = "./tests/data/50Hz/testfile1_50Hz.DAT"
    rec = comtrade.load(filename_cfg, filename_dat)

    freq = rec.cfg.frequency  # system frequency
    srate = rec.cfg.sample_rates[0][0]  # sampling rate
    N = int(srate / freq)  # Number of sample per cycle

    UL1 = np.array(rec.analog[0])
    IL1 = np.array(rec.analog[1])

    UL1_dft = dft(UL1, N)
    IL1_dft = dft(IL1, N)

    UL1_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile1_50Hz_UL1_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile1_50Hz_IL1_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_UL1_ang = np.loadtxt(
        "./tests/results/50Hz/testfile1_50Hz_IL1_UL1_ang.csv",
        delimiter=",",
        dtype=np.float64,
    )

    # np.allclose(A,B,...)
    assert np.array_equal(abs(UL1_dft), UL1_dft_res)
    assert np.array_equal(abs(IL1_dft), IL1_dft_res)
    assert np.array_equal(np.angle(IL1_dft / UL1_dft), IL1_UL1_ang)


# Associated with testfile2_50Hz.cfg
def test_dft_2():
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

    UL1_dft = dft(UL1, N)
    UL2_dft = dft(UL2, N)
    UL3_dft = dft(UL3, N)
    IL1_dft = dft(IL1, N)
    IL2_dft = dft(IL2, N)
    IL3_dft = dft(IL3, N)

    UL1_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_UL1_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL2_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_UL2_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL3_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_UL3_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL1_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL2_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL2_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL3_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL3_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_UL1_ang = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL1_UL1_ang.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL2_UL2_ang = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL2_UL2_ang.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL3_UL3_ang = np.loadtxt(
        "./tests/results/50Hz/testfile2_50Hz_IL3_UL3_ang.csv",
        delimiter=",",
        dtype=np.float64,
    )

    # np.allclose(A,B,...)
    assert np.array_equal(abs(UL1_dft), UL1_dft_res)
    assert np.array_equal(abs(UL2_dft), UL2_dft_res)
    assert np.array_equal(abs(UL3_dft), UL3_dft_res)
    assert np.array_equal(abs(IL1_dft), IL1_dft_res)
    assert np.array_equal(abs(IL2_dft), IL2_dft_res)
    assert np.array_equal(abs(IL3_dft), IL3_dft_res)
    assert np.array_equal(np.angle(IL1_dft / UL1_dft), IL1_UL1_ang)
    assert np.array_equal(np.angle(IL2_dft / UL2_dft), IL2_UL2_ang)
    assert np.array_equal(np.angle(IL3_dft / UL3_dft), IL3_UL3_ang)


# Associated with testfile3_50Hz.cfg
def test_dft_3():
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

    UL1_dft = dft(UL1, N)
    UL2_dft = dft(UL2, N)
    UL3_dft = dft(UL3, N)
    IL1_dft = dft(IL1, N)
    IL2_dft = dft(IL2, N)
    IL3_dft = dft(IL3, N)

    UL1_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_UL1_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL2_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_UL2_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    UL3_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_UL3_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL1_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL2_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL2_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL3_dft_res = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL3_dft.csv",
        delimiter=",",
        dtype=np.float64,
    )

    IL1_UL1_ang = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL1_UL1_ang.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL2_UL2_ang = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL2_UL2_ang.csv",
        delimiter=",",
        dtype=np.float64,
    )
    IL3_UL3_ang = np.loadtxt(
        "./tests/results/50Hz/testfile3_50Hz_IL3_UL3_ang.csv",
        delimiter=",",
        dtype=np.float64,
    )

    # np.allclose(A,B,...)
    assert np.array_equal(abs(UL1_dft), UL1_dft_res)
    assert np.array_equal(abs(UL2_dft), UL2_dft_res)
    assert np.array_equal(abs(UL3_dft), UL3_dft_res)
    assert np.array_equal(abs(IL1_dft), IL1_dft_res)
    assert np.array_equal(abs(IL2_dft), IL2_dft_res)
    assert np.array_equal(abs(IL3_dft), IL3_dft_res)
    assert np.array_equal(np.angle(IL1_dft / UL1_dft), IL1_UL1_ang)
    assert np.array_equal(np.angle(IL2_dft / UL2_dft), IL2_UL2_ang)
    assert np.array_equal(np.angle(IL3_dft / UL3_dft), IL3_UL3_ang)
