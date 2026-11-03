"""
PSDs from tabulated amplitude spectral densities (ASD).

- LIGO O5c design sensitivity, file PSD_O5.txt, see https://dcc.ligo.org/LIGO-T2500310
  Columns: [Frequency (Hz), O5aStrain, O5bStrain, O5cStrain], ASD in units of 1/sqrt(Hz).
  The O5c column is used.

- Einstein Telescope, one L-shaped detector (90 deg opening angle), cryogenic xylophone design,
  see https://gitlab.et-gw.eu/pub/et-sensitivity-curves
  Each detector is made of two interferometers, LF and HF, with one 'total' file each, to be
  placed in ET_sensitivity/ as totalLF.txt and totalHF.txt.
  Columns: [Frequency (Hz), displacement ASD (m/sqrt(Hz)), strain ASD with high-frequency
  correction for a signal 15 deg off zenith, strain ASD with high-frequency correction for a
  signal at zenith]. The strain ASD at zenith (column index 3) is used.
"""
import numpy as np
import os

_DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PSD_O5.txt")
_ET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ET_sensitivity")
_DEFAULT_ET_LF_PATH = os.path.join(_ET_DIR, "totalLF.txt")
_DEFAULT_ET_HF_PATH = os.path.join(_ET_DIR, "totalHF.txt")


def _psd_from_table(freq_tab, psd_tab, num_points, delta_f, f_low):
    """
    Log-log interpolation of a tabulated PSD on the uniform frequency grid
    f_k = k * delta_f, k = 0, ..., num_points - 1 (the same grid as the waveforms).
    Frequencies below f_low and outside the tabulated range are set to np.inf, so they
    carry zero weight in a noise-weighted inner product (1/inf = 0).
    """
    freqs = np.arange(num_points) * delta_f

    # Log-log interpolation
    with np.errstate(divide="ignore"):   # log(0) = -inf at f = 0, without a warning
        log_f = np.log(freqs)
    log_psd = np.interp(
        log_f, np.log(freq_tab), np.log(psd_tab),
        left=np.inf, right=np.inf
    )
    psd = np.exp(log_psd)

    # Zero weight below f_low and outside the tabulated frequency range
    psd[freqs < f_low] = np.inf
    psd[freqs < freq_tab[0]] = np.inf
    psd[freqs > freq_tab[-1]] = np.inf

    return psd


def get_psd_O5(num_points, delta_f, f_low, data_path=_DEFAULT_DATA_PATH):
    """
    O5c design PSD of one LIGO detector.

    Parameters
    ----------
    num_points : int        - length of the frequency series (same as that of the waveforms)
    delta_f    : float [Hz] - frequency resolution
    f_low      : float [Hz] - starting frequency of the waveform
    data_path  : str        - path to the tabulated O5 ASD data file

    Returns
    -------
    psd : np.ndarray [1/Hz] (with length num_points)
    """
    data = np.loadtxt(data_path, skiprows=1)
    freq_tab = data[:, 0]
    asd_O5c = data[:, 3]
    psd_tab = asd_O5c ** 2  # ASD [1/sqrt(Hz)] -> PSD [1/Hz]
    return _psd_from_table(freq_tab, psd_tab, num_points, delta_f, f_low)


def get_psd_ET(num_points, delta_f, f_low, lf_path=_DEFAULT_ET_LF_PATH,
               hf_path=_DEFAULT_ET_HF_PATH, column=3):
    """
    PSD of one L-shaped ET detector: the LF and HF interferometers measure the same strain
    with independent noises, so their PSDs combine as 1/S = 1/S_LF + 1/S_HF.

    Parameters
    ----------
    num_points : int        - length of the frequency series (same as that of the waveforms)
    delta_f    : float [Hz] - frequency resolution
    f_low      : float [Hz] - starting frequency of the waveform
    lf_path    : str        - path to the LF 'total' file
    hf_path    : str        - path to the HF 'total' file
    column     : int (optional), default = 3 - column of the strain ASD (3 = signal at zenith)

    Returns
    -------
    psd : np.ndarray [1/Hz] (with length num_points)
    """
    inverse_psd = np.zeros(num_points)
    for path in (lf_path, hf_path):
        data = np.loadtxt(path)
        psd = _psd_from_table(data[:, 0], data[:, column] ** 2, num_points, delta_f, f_low)
        inverse_psd += 1.0 / psd    # 1/inf = 0 outside the band of each interferometer
    with np.errstate(divide="ignore"):
        return 1.0 / inverse_psd    # inf where neither interferometer is sensitive
