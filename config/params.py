"""
Global parameters of the project (physical constants, frequencies, injection settings,
priors and sampler settings).

The injection settings follow arXiv:2402.15336.
Edit this file to change the simulation settings: every script reads its defaults from here.
"""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Physical constants (SI units, same values as LAL)
MSUN_SI   = 1.9884099021470415e30   # kg
G_SI      = 6.6743e-11              # m^3 kg^-1 s^-2
C_SI      = 2.99792458e8            # m/s
MPC_SI    = 3.085677581491367e+22   # m

MSUN_SEC  = G_SI * MSUN_SI / C_SI**3   # solar mass in seconds, ~ 4.9254910255e-6 s
MPC_SEC   = MPC_SI / C_SI              # megaparsec in seconds,  ~ 1.029e14 s

# Frequencies
F_LOW     = 20.0              # Hz - minimum analysis frequency
F_REF     = 20.0              # Hz - reference frequency at which the phase is referenced
F_HIGH    = 1024.0            # Hz - Nyquist frequency / PSD upper limit

# Detector network (zero-noise O5 runs and real-noise O4b runs)
IFO_LIST  = ["H1", "L1"]

# Folder with the GWOSC O4b 4 kHz frames (.gwf), with one subfolder per detector (H1/, L1/),
# used by the real-noise runs and by make_segments.py. Set it to your local copy, or set the
# environment variable O4B_DATA_DIR (e.g. export O4B_DATA_DIR=/path/to/public-O4b).
O4B_DATA_DIR = os.environ.get("O4B_DATA_DIR", os.path.join(_REPO_ROOT, "data", "O4b"))
# O4B_DATA_DIR = "/home/shared/virgo/data/public-O4b"

# Fixed injection parameters (Sec. IV of the paper)
M_TOTAL_SIM   = 32.0          # M_sun - injected total mass
RA_SIM        = 0.0           # rad   - right ascension
DEC_SIM       = 0.0           # rad   - declination
PSI_SIM       = 0.0           # rad   - polarization angle
IOTA_SIM      = 0.5           # rad   - inclination angle
GEOCENT_TIME  = 1187008882.4  # s     - GPS reference time of the zero-noise injections (arbitrary)

# Mass ratios q = m_1/m_2
Q_MASS_RATIO = 3.0                               # Figs. 2, 4, 5 and the ET run
Q_MASS_RATIO_CONFIGS_FIG3 = [1.15, 2.0, 3.0, 4.0]

# Aligned spins (chi_1, chi_2)
CHI_1, CHI_2 = 0.7, 0.6                          # Figs. 3, 4, 5 and the ET run
SPIN_CONFIGS_FIG2 = [(0.7, 0.6), (0.7, 0.1), (0.5, 0.3), (0.2, 0.1)]

# Target network SNR
TARGET_SNR = 120                                 # Figs. 2, 3, 4
TARGET_SNR_FIG5 = [260, 130, 80, 40]             # Fig. 5

# Injected charges
Q_INJECTED_FIG4 = [0.1, 0.3, 0.5, 0.7]           # Fig. 4

# Bilby priors (all uniform)
PRIOR_Q_BH    = (-1.0,  1.0)  # braneworld charge
PRIOR_M1      = (5.0,  50.0)  # M_sun
PRIOR_M2      = (5.0,  50.0)  # M_sun
PRIOR_CHI     = (0.0,  0.99)  # aligned spin component

# Sampler
SAMPLER        = "dynesty"
SAMPLER_KWARGS = {
    "nlive": 500,
    "dlogz": 0.1,
    "sample": "acceptance-walk",
    "naccept": 30,
    "bound": "live",
    "npool": 32,              # number of parallel processes: set it to the cores available
}