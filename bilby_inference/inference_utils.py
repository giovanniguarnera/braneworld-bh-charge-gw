"""
Utilities for the injection and parameter estimation runs with bilby.

Three detector settings are supported by run_injection:
    - zero noise, LIGO H1 + L1 with the O5c design PSD (reproduction of arXiv:2402.15336);
    - real O4b noise, H1 + L1, with the PSD measured around the injection time (GWOSC data);
    - zero noise, Einstein Telescope in the 2L configuration with its design PSD.
"""

import sys
import os
import glob
import re
from functools import partial

BILBY_DIR = os.path.dirname(os.path.abspath(__file__))   # .../bilby_inference
PROJECT_DIR = os.path.dirname(BILBY_DIR)                 # repository root
sys.path.insert(0, BILBY_DIR)
sys.path.insert(0, PROJECT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import bilby

from source_model import TaylorF2_charged_source_model
from psd import get_psd_O5, get_psd_ET
from config.params import (
    F_LOW, F_HIGH, IFO_LIST, GEOCENT_TIME, RA_SIM, DEC_SIM, PSI_SIM, IOTA_SIM,
    PRIOR_M1, PRIOR_M2, PRIOR_CHI, PRIOR_Q_BH, SAMPLER, SAMPLER_KWARGS, O4B_DATA_DIR,
)

DELTA_F = 1.0 / 32.0               # Hz
DURATION = 1.0 / DELTA_F           # s
SAMPLING_FREQUENCY = 2.0 * F_HIGH  # Hz - twice the Nyquist frequency

# Real-noise injections (O4b, GWOSC 4 kHz frames, folder O4B_DATA_DIR in config/params.py)
# - same values as scripts/make_segments.py
POST_TRIGGER = 2.0          # s - analysis segment ends 2 s after t_c (as in the zero-noise runs)
PSD_DURATION = 1024.0       # s - off-source stretch before the analysis segment, for the PSD
DATA_PAD = 8.0              # s - read on each side, cropped away after resampling
TUKEY_ROLL_OFF = 0.4        # s - Tukey window roll-off, same for PSD and strain (bilby_pipe default)
MAX_FREQUENCY_REAL = 0.875 * SAMPLING_FREQUENCY / 2   # Hz - below the resampling filter roll-off

# Einstein Telescope, 2L configuration (arXiv:2303.15923): two L-shaped detectors with 15 km
# arms, one in Sardinia and one in the Meuse-Rhine region, rotated by 45 deg relative to each other.
#
# Coordinates and orientations are taken from the detector list of the gwfast code
# (entries 'ETSL' and 'ETMRL45d').
#
# Orientation: for an L-shaped detector one angle is enough, since the arms are at 90 deg.
# 'bisector' is the direction halfway between the two arms, measured from East towards North.
#   Sardinia:    bisector = 45 deg -> arms pointing East (0 deg) and North (90 deg)
#   Meuse-Rhine: bisector = 90 deg -> arms pointing North-East (45 deg) and North-West (135 deg)
# bilby wants the direction of each arm (same convention, from East towards North):
#   xarm_azimuth = bisector - 45,  yarm_azimuth = bisector + 45
ET_2L_SITES = {
    'ETS':  dict(latitude=40.0 + 31.0 / 60.0, longitude=9.0 + 25.0 / 60.0, bisector=45.0),
    'ETMR': dict(latitude=50.0 + 43.0 / 60.0 + 23.0 / 3600.0,
                 longitude=5.0 + 55.0 / 60.0 + 14.0 / 3600.0, bisector=90.0),
}
ET_ARM_LENGTH = 15.0        # km
DURATION_ET = 256.0         # s - contains the inspiral from F_LOW_ET (~146 s for 24 + 8 Msun)
F_LOW_ET = 5.0              # Hz - below this the gain in SNR is negligible (~3%)


def masses_from_q(q, M_total):
    """
    Parameters
    ----------
    q       : float         - mass ratio, q = m_1/m_2 >= 1
    M_total : float [M_sun] - total mass

    Returns
    -------
    m_1, m_2  : float [M_sun] - component masses
    """
    m_2 = M_total / (1.0 + q)
    m_1 = M_total - m_2
    return m_1, m_2


def make_waveform_generator(duration=DURATION, f_low=F_LOW):
    """
    Parameters
    ----------
    duration : float [s] (optional), default = DURATION - length of the analysis segment
    f_low    : float [Hz] (optional), default = F_LOW   - starting frequency of the waveform

    Returns
    -------
    waveform_generator : bilby.gw.WaveformGenerator - wraps TaylorF2_charged_source_model
    """
    return bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=TaylorF2_charged_source_model,
        waveform_arguments={'minimum_frequency': f_low},
    )


def make_ifo_network(frequency_array):
    """
    H1 + L1 network with the O5c design PSD and zero-noise strain.

    Parameters
    ----------
    frequency_array : np.ndarray [Hz] - frequency grid of the waveform generator

    Returns
    -------
    ifos : bilby InterferometerList - detector network with the O5 PSD and zero-noise strain
    """
    num_points = len(frequency_array)
    psd = get_psd_O5(num_points, DELTA_F, F_LOW)
    psd_data = np.asarray(psd, dtype=float)
    psd_data = np.where(np.isinf(psd_data), 1e10, psd_data)  # bilby does not handle np.inf

    ifos = bilby.gw.detector.InterferometerList(IFO_LIST)
    for ifo in ifos:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequency_array, psd_array=psd_data)
        ifo.minimum_frequency = F_LOW
        ifo.maximum_frequency = frequency_array[-1]
        ifo.duration = DURATION
        ifo.sampling_frequency = SAMPLING_FREQUENCY
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION,
            start_time=GEOCENT_TIME - DURATION + POST_TRIGGER)
    return ifos


def make_ifo_network_ET(frequency_array):
    """
    Same role as make_ifo_network, for the Einstein Telescope 2L network (see ET_2L_SITES):
    the two L-shaped detectors are not predefined in bilby, so they are built here.

    Parameters
    ----------
    frequency_array : np.ndarray [Hz] - frequency grid of the waveform generator

    Returns
    -------
    ifos : bilby InterferometerList - ET 2L network with the ET PSD and zero-noise strain
    """
    num_points = len(frequency_array)
    psd = get_psd_ET(num_points, 1.0 / DURATION_ET, F_LOW_ET)
    psd_data = np.where(np.isinf(psd), 1e10, psd)  # bilby does not handle np.inf

    ifos = bilby.gw.detector.InterferometerList([])
    for name, site in ET_2L_SITES.items():
        ifo = bilby.gw.detector.Interferometer(
            name=name,
            power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
                frequency_array=frequency_array, psd_array=psd_data),
            minimum_frequency=F_LOW_ET, maximum_frequency=frequency_array[-1],
            length=ET_ARM_LENGTH, latitude=site['latitude'], longitude=site['longitude'],
            elevation=0.0,  # underground (~ -200 m): negligible for the antenna patterns
            xarm_azimuth=site['bisector'] - 45.0, yarm_azimuth=site['bisector'] + 45.0,
            xarm_tilt=0.0, yarm_tilt=0.0,
        )
        ifo.duration = DURATION_ET
        ifo.sampling_frequency = SAMPLING_FREQUENCY
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION_ET,
            start_time=GEOCENT_TIME - DURATION_ET + POST_TRIGGER)
        ifos.append(ifo)
    return ifos


_FNAME_GPS = re.compile(r"-(\d+)-(\d+)\.gwf$")


def load_strain(ifo, start, end):
    """
    Read the O4b strain of one detector from the GWOSC frames in O4B_DATA_DIR.

    Parameters
    ----------
    ifo        : str       - detector name ('H1', 'L1')
    start, end : float [s] - GPS interval

    Returns
    -------
    data : gwpy TimeSeries - O4b strain at the native 4096 Hz
    """
    import matplotlib as mpl
    with mpl.rc_context():   # importing gwpy restyles matplotlib globally (grid, fonts, ...), we undo it
        from gwpy.timeseries import TimeSeries

    # Snap the interval outwards to the 4096 Hz sample grid of the frames (integer GPS + k/4096 s):
    # FrameL reads whole samples, so a non-aligned interval would come back one sample short
    frame_rate = 4096
    start = np.floor(start * frame_rate) / frame_rate
    end = np.ceil(end * frame_rate) / frame_rate

    files = []
    for path in glob.glob(os.path.join(O4B_DATA_DIR, ifo, "*.gwf")):
        match = _FNAME_GPS.search(os.path.basename(path))
        f_start, f_dur = int(match.group(1)), int(match.group(2))
        if f_start < end and f_start + f_dur > start:
            files.append((f_start, f_dur, path))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No {ifo} frames covering [{start}, {end}) in {O4B_DATA_DIR}")

    # Read from each file only the part it contains, then join: passing the whole
    # interval to every file makes FrameL return full-length pieces that cannot be joined
    channel = f"{ifo}:GWOSC-4KHZ_R1_STRAIN"
    data = None
    for f_start, f_dur, path in files:
        piece = TimeSeries.read(path, channel, start=max(start, f_start),
                                end=min(end, f_start + f_dur), format="gwf.framel")
        data = piece if data is None else data.append(piece, inplace=False)

    if not np.all(np.isfinite(data.value)):
        raise ValueError(f"{ifo}: NaN in [{start}, {end}) - is this time in usable.json?")
    return data


def _crop_exact(data, start, duration, sampling_frequency):
    """Exactly duration * sampling_frequency samples of `data`, starting at GPS `start`."""
    i0 = int(round((start - data.t0.value) * sampling_frequency))
    n = int(round(duration * sampling_frequency))
    out = data[i0:i0 + n]
    if len(out) != n:
        raise ValueError("Not enough data to crop the requested stretch")
    return out


def make_ifo_network_real(geocent_time):
    """
    Same role as make_ifo_network, with O4b data instead of the O5 design curve:
      - PSD    : median Welch estimate on the PSD_DURATION seconds preceding the analysis segment
      - strain : the actual O4b strain in the analysis segment [t_c + 2 - DURATION, t_c + 2]

    Parameters
    ----------
    geocent_time : float [s] - GPS injection time (take it from segments/injection_times.json)

    Returns
    -------
    ifos : bilby InterferometerList - detector network with measured PSD and real-noise strain
    """
    analysis_end = geocent_time + POST_TRIGGER
    analysis_start = analysis_end - DURATION
    psd_start = analysis_start - PSD_DURATION
    tukey_alpha = 2.0 * TUKEY_ROLL_OFF / DURATION

    ifos = bilby.gw.detector.InterferometerList(IFO_LIST)
    for ifo in ifos:
        raw = load_strain(ifo.name, psd_start - DATA_PAD, analysis_end + DATA_PAD)
        data = raw.resample(SAMPLING_FREQUENCY)   # 4096 -> 2048 Hz, anti-aliased

        # PSD: fftlength = DURATION so that its frequency grid is the waveform one (delta_f = 1/32 Hz)
        psd_data = _crop_exact(data, psd_start, PSD_DURATION, SAMPLING_FREQUENCY)
        psd = psd_data.psd(fftlength=DURATION, overlap=DURATION / 2,
                           window=("tukey", tukey_alpha), method="median")
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value)

        # Strain: the real noise the signal will be injected into
        ifo.strain_data.roll_off = TUKEY_ROLL_OFF
        ifo.set_strain_data_from_gwpy_timeseries(
            _crop_exact(data, analysis_start, DURATION, SAMPLING_FREQUENCY))

        ifo.minimum_frequency = F_LOW
        ifo.maximum_frequency = MAX_FREQUENCY_REAL
    return ifos


def network_snr(ifos, waveform_generator, params):
    """
    Parameters
    ----------
    ifos               : bilby InterferometerList - detector network
    waveform_generator : bilby WaveformGenerator  - waveform generator
    params             : dict                     - waveform + extrinsic parameters

    Returns
    -------
    snr : float - network optimal SNR (quadrature sum over detectors)
    """
    hf_signal = waveform_generator.frequency_domain_strain(params)
    snr_sq = sum(np.real(ifo.optimal_snr_squared(signal=ifo.get_detector_response(hf_signal, params)))
                 for ifo in ifos)
    return float(snr_sq) ** 0.5


def scale_distance_for_snr(ifos, waveform_generator, params, target_snr, trial_distance=500.0):
    """
    Luminosity distance giving a target network SNR. The strain amplitude scales exactly as
    1/d_L, so one SNR evaluation at a trial distance fixes the distance for any target SNR.

    Parameters
    ----------
    ifos               : bilby InterferometerList - detector network
    waveform_generator : bilby WaveformGenerator  - waveform generator
    params             : dict                     - injection parameters (distance is overridden)
    target_snr         : float                    - desired network SNR
    trial_distance     : float [Mpc] (optional), default = 500.0 - distance of the trial evaluation

    Returns
    -------
    distance : float [Mpc] - luminosity distance giving the target network SNR
    """
    trial_params = dict(params, luminosity_distance=trial_distance)
    snr_trial = network_snr(ifos, waveform_generator, trial_params)
    return trial_distance * snr_trial / target_snr


def constraint_conversion(parameters, fixed_values=None):
    """
    Add the quantities the constraint priors act on. Each block is independent and is
    skipped when its own inputs are not available.

    delta_mass = mass_1 - mass_2 enforces mass_1 >= mass_2.
    horizon_i  = 1 - chi_i^2 + q_BH (M/m_i)^2 enforces the existence of the i-th horizon
                 (only used when run_injection is called with horizon_constraint=True).

    Parameters
    ----------
    parameters   : dict           - sampled parameters
    fixed_values : dict, optional - values of the parameters that are not sampled

    Returns
    -------
    converted_parameters : dict - parameters plus delta_mass and horizon_1, horizon_2
    """
    converted_parameters = parameters.copy()
    if fixed_values is not None:
        for key, value in fixed_values.items():
            converted_parameters.setdefault(key, value)

    masses = ('mass_1', 'mass_2')

    if all(key in converted_parameters for key in masses):
        converted_parameters['delta_mass'] = (
            converted_parameters['mass_1'] - converted_parameters['mass_2'])

    if all(key in converted_parameters for key in masses + ('chi_1', 'chi_2', 'q_BH')):
        M = converted_parameters['mass_1'] + converted_parameters['mass_2']
        for i in (1, 2):
            converted_parameters[f'horizon_{i}'] = (
                1.0 - converted_parameters[f'chi_{i}']**2
                + converted_parameters['q_BH'] * (M / converted_parameters[f'mass_{i}'])**2)

    return converted_parameters


def run_injection(m_1, m_2, chi_1, chi_2, q_BH_injected, target_snr,
                  label, outdir, free_masses_and_spins=True,
                  horizon_constraint=False, sampler_kwargs=None,
                  time_marginalization=True, phase_marginalization=True,
                  time_prior_width=0.1, injection_time=None,
                  uncharged_model=False, einstein_telescope=False):
    """
    Run a single injection + parameter estimation, in zero noise (default) or real O4b noise.

    Parameters
    ----------
    m_1, m_2              : float [M_sun]     - injected component masses (m_1 >= m_2)
    chi_1, chi_2          : float (with sign) - injected aligned spins
    q_BH_injected         : float (with sign) - injected braneworld charge
    target_snr            : float             - network SNR the distance is scaled to
    label                 : str               - bilby run label (used for output filenames)
    outdir                : str               - output directory
    free_masses_and_spins : bool (optional), default = True - if False, only q_BH is sampled
                            (fast 1D validation run)
    horizon_constraint    : bool (optional), default = False - if True, restrict the
                            sampling to 1 - chi_i^2 + q_BH (M/m_i)^2 >= 0 (i = 1, 2),
                            i.e. both horizons exist
    sampler_kwargs        : dict (optional), default = None - overrides SAMPLER_KWARGS
                            (useful on clusters, to match npool to the allocated cores)
    time_marginalization  : bool (optional), default = True - analytically marginalize
                            over geocent_time; geocent_time is given a Uniform prior of
                            half-width `time_prior_width` around its injected value and
                            reconstructed in the posterior afterwards.
    phase_marginalization : bool (optional), default = True - analytically marginalize
                            over the coalescence phase
    time_prior_width      : float [s] (optional), default = 0.1 - half-width of the
                            Uniform prior on geocent_time used for time marginalization
                            (ignored if time_marginalization=False)
    injection_time        : float [s] (optional), default = None - if None, zero noise with
                            the O5 design PSD at GEOCENT_TIME (paper reproduction), otherwise
                            GPS time of an injection into real O4b noise, with the PSD measured
                            there (take it from segments/injection_times.json)
    uncharged_model       : bool (optional), default = False - if False, q_BH is sampled with
                            the PRIOR_Q_BH prior, otherwise q_BH is fixed to 0 for the inference,
                            to measure the bias on masses and spins when the charge is ignored
    einstein_telescope    : bool (optional), default = False - if True, zero noise with the
                            Einstein Telescope 2L network and design PSD (F_LOW_ET, DURATION_ET)
                            instead of O5; cannot be combined with injection_time

    Returns
    -------
    result : bilby result object - posterior samples, evidence and run metadata
    """
    if sampler_kwargs is None:
        sampler_kwargs = SAMPLER_KWARGS

    if einstein_telescope:
        if injection_time is not None:
            raise ValueError("einstein_telescope is a zero-noise run: injection_time must be None")
        geocent_time = GEOCENT_TIME
        waveform_generator = make_waveform_generator(duration=DURATION_ET, f_low=F_LOW_ET)
        ifos = make_ifo_network_ET(waveform_generator.frequency_array)
    elif injection_time is None:
        geocent_time = GEOCENT_TIME
        waveform_generator = make_waveform_generator()
        ifos = make_ifo_network(waveform_generator.frequency_array)
    else:
        geocent_time = injection_time
        waveform_generator = make_waveform_generator()
        ifos = make_ifo_network_real(geocent_time)

    # bilby requires the standard parameter names (mass_1, mass_2, ...) as dict keys
    base_params = dict(
        mass_1=m_1, mass_2=m_2, chi_1=chi_1, chi_2=chi_2, q_BH=q_BH_injected,
        iota=IOTA_SIM, phase=0.0, ra=RA_SIM, dec=DEC_SIM, psi=PSI_SIM,
        geocent_time=geocent_time,
    )
    distance = scale_distance_for_snr(ifos, waveform_generator, base_params, target_snr)
    injection_parameters = dict(base_params, luminosity_distance=distance)

    for ifo in ifos:
        ifo.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

    fixed_values = dict(mass_1=m_1, mass_2=m_2, chi_1=chi_1, chi_2=chi_2)
    if uncharged_model:
        if not free_masses_and_spins:
            raise ValueError("uncharged_model needs free_masses_and_spins=True: nothing left to sample")
        fixed_values['q_BH'] = 0.

    # Priors: q_BH free unless the uncharged model is used; masses and spins free only in the full analysis
    priors = bilby.core.prior.PriorDict(
        conversion_function=partial(constraint_conversion, fixed_values=fixed_values))
    if uncharged_model:
        priors['q_BH'] = 0.
    else:
        priors['q_BH'] = bilby.core.prior.Uniform(*PRIOR_Q_BH, name='q_BH', latex_label='$q^{BH}$')

    if horizon_constraint:
        for i in (1, 2):
            priors[f'horizon_{i}'] = bilby.core.prior.Constraint(
                minimum=0.0, maximum=np.inf, name=f'horizon_{i}')

    if free_masses_and_spins:
        priors['mass_1'] = bilby.core.prior.Uniform(*PRIOR_M1, name='mass_1', latex_label='$m_1$')
        priors['mass_2'] = bilby.core.prior.Uniform(*PRIOR_M2, name='mass_2', latex_label='$m_2$')
        priors['chi_1'] = bilby.core.prior.Uniform(*PRIOR_CHI, name='chi_1', latex_label='$\\chi_1$')
        priors['chi_2'] = bilby.core.prior.Uniform(*PRIOR_CHI, name='chi_2', latex_label='$\\chi_2$')
        priors['delta_mass'] = bilby.core.prior.Constraint(
            minimum=0, maximum=PRIOR_M1[1] - PRIOR_M2[0], name='delta_mass')
        fixed_keys = ['luminosity_distance', 'iota', 'phase', 'ra', 'dec', 'psi', 'geocent_time']
    else:
        fixed_keys = ['mass_1', 'mass_2', 'chi_1', 'chi_2',
                      'luminosity_distance', 'iota', 'phase', 'ra', 'dec', 'psi', 'geocent_time']

    if time_marginalization:
        fixed_keys = [key for key in fixed_keys if key != 'geocent_time']
        priors['geocent_time'] = bilby.core.prior.Uniform(
            minimum=injection_parameters['geocent_time'] - time_prior_width,
            maximum=injection_parameters['geocent_time'] + time_prior_width,
            name='geocent_time', latex_label='$t_c$')
    if phase_marginalization:
        fixed_keys = [key for key in fixed_keys if key != 'phase']
        priors['phase'] = bilby.core.prior.Uniform(
            minimum=0, maximum=2 * np.pi, boundary='periodic',
            name='phase', latex_label='$\\phi_c$')

    for key in fixed_keys:
        priors[key] = injection_parameters[key]

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator, priors=priors,
        time_marginalization=time_marginalization, phase_marginalization=phase_marginalization,
    )

    result = bilby.run_sampler(
        likelihood=likelihood, priors=priors, sampler=SAMPLER,
        outdir=outdir, label=label, injection_parameters=injection_parameters,
        resume=True, plot=False,
        **sampler_kwargs,
    )
    return result


def plot_posteriors(results, save_path, injected_values=None, xlabel=r"$q^{\mathrm{BH}}$",
                    prior_range=(-1.0, 1.0), n_grid=500, show_ci=True, level=68.0):
    """
    Show the q_BH posteriors as KDE curves over the full prior range, with the bounds of the
    equal-tailed `level`% credible interval as dashed lines. Median and credible interval of
    each run are also printed.

    Parameters
    ----------
    results         : dict {legend label: bilby result} - runs to overlay
    save_path       : str                    - path of the output figure
    injected_values : list of float, optional - injected charges, drawn as dotted lines
    xlabel          : str, optional          - label of the x axis
    prior_range     : tuple, optional        - x range of the plot (the q_BH prior)
    n_grid          : int, optional          - number of points of the KDE grid
    show_ci         : bool, optional         - if True, draw the credible bounds
    level           : float [%], optional    - credible level

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """
    from scipy.stats import gaussian_kde

    grid = np.linspace(prior_range[0], prior_range[1], n_grid)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for legend_label, result in results.items():
        samples = np.asarray(result.posterior['q_BH'])
        kde = gaussian_kde(samples)
        line, = ax.plot(grid, kde(grid), lw=1.8, label=legend_label)

        low, high = np.percentile(samples, [50 - level / 2, 50 + level / 2])
        print(f"{legend_label}: median = {np.median(samples):+.3f}, "
              f"{level:g}% CI = [{low:+.3f}, {high:+.3f}], width = {high - low:.3f}")

        if show_ci:
            for bound in (low, high):
                ax.axvline(bound, color=line.get_color(), ls='--', lw=0.9, alpha=0.7)

    if injected_values:
        for value in injected_values:
            ax.axvline(value, color='k', ls=':', lw=1.0)

    ax.set_xlim(*prior_range)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig, ax