# braneworld-bh-charge-gw

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23925540.svg)](https://doi.org/10.5281/zenodo.23925540)

Code for the master's thesis *Constraining the braneworld charge of black
holes using gravitational waves* (Giovanni Guarnera, University of Bologna, 2026).

In a braneworld scenario, black holes on the brane can carry a tidal charge
$Q^{\rm BH}$ inherited from the extra dimension. The charge changes the horizon of each black
hole and therefore the energy and angular momentum it absorbs during the inspiral
(tidal heating), which leaves a phase correction $\delta\psi(f)$ in the gravitational-wave
signal. Following [arXiv:2402.15336](https://arxiv.org/abs/2402.15336), this repository

- implements a TaylorF2 waveform with the tidal heating phase correction as a function of the
  dimensionless braneworld charge $q^{\rm BH} = Q^{\rm BH}/M^2$;
- runs injection and Bayesian parameter estimation studies with
  [bilby](https://git.ligo.org/lscsoft/bilby) and dynesty, to measure how well $q^{\rm BH}$ is
  constrained:
  - in zero noise with the LIGO O5c design sensitivity (as in arXiv:2402.15336);
  - in real LIGO O4b noise from [GWOSC](https://gwosc.org), with the PSD measured around each
    injection, extending the analysis;
  - in zero noise with the Einstein Telescope (2L configuration).

## Repository layout

```
.
├── config/params.py              # all settings: constants, injections, priors, sampler
├── TaylorF2.py                   # point-particle TaylorF2 (3.5PN phase, restricted amplitude)
├── tidal_heating_correction.py   # tidal heating phase correction delta_psi(f; q_BH)
├── TaylorF2_charged.py           # TaylorF2 + delta_psi, band-limited to [f_low, 2 f_ISCO]
├── isco.py                       # ISCO of the remnant and cutoff frequency
├── psd.py                        # O5c and Einstein Telescope PSDs
├── plotting.py                   # mismatch-vs-charge plot
├── PSD_O5.txt                    # LIGO O5 design ASDs (LIGO-T2500310)
├── ET_sensitivity/               # ET sensitivity curves (to download, see below)
├── *_test.ipynb                  # tests of the waveform modules
├── scripts/
│   ├── make_fig1.py              # mismatch vs q_BH (Fig. 1), saved to img/
│   └── make_fig1_test.ipynb      # runs make_fig1.py from a notebook
└── bilby_inference/
    ├── source_model.py           # bilby frequency-domain source model
    ├── inference_utils.py        # detector networks, SNR scaling, priors, run_injection
    ├── segments/                 # O4b injection times (written by make_segments.py)
    └── scripts/
        ├── make_fig2.py ... make_fig5.py   # injection + inference runs of Figs. 2-5
        ├── make_fig_ET.py                  # Einstein Telescope run
        ├── make_segments.py                # O4b segment lists and injection times
        ├── diagnostic_corner.py            # corner plots + credible intervals of a folder
        ├── shared_corner_plots.py          # overlaid corner plots of two sets of runs
        ├── check_real_noise.ipynb          # ASD and whitened strain of one O4b realization
        └── compare_zero_vs_O4b.ipynb       # medians and 68% intervals, zero noise vs O4b
```

## Installation

The code was developed and run with Python 3.10 in a conda environment:

```bash
conda create -n braneworld python=3.10
conda activate braneworld
conda install -c conda-forge gsl          # needed to build pyseobnr
pip install -r requirements.txt
```

`pycbc` is needed only for Fig. 1 (`scripts/make_fig1.py`).

`requirements.txt` lists only the packages imported by the code. The full environment used
for the thesis runs (`pip freeze` output, Linux, Python 3.10) is in `requirements-freeze.txt`.

All commands and imports assume that you work from the repository root (the notebooks in
`scripts/` and `bilby_inference/scripts/` are the exception: open them from their folder).

## Quick start: generating a waveform

```python
import numpy as np
from TaylorF2_charged import TaylorF2_charged

delta_f = 1 / 32                   # Hz

hp, hc = TaylorF2_charged(
    q_BH=0.5,                      # braneworld charge, in [-1, 1]
    mass_1=24.0, mass_2=8.0,       # M_sun, mass_1 >= mass_2
    chi_1=0.7, chi_2=0.6,          # aligned spins
    distance=500.0,                # Mpc
    inclination=0.5, phase=0.0,
    delta_f=delta_f, f_low=20.0, f_high=1024.0, f_ref=20.0,
)
freqs = np.arange(len(hp)) * delta_f   # frequency grid of hp, hc: f_k = k * delta_f, from 0 Hz
```

`th_on=False` gives the point-particle TaylorF2 without tidal heating. The tests of the
waveform modules are in `isco_test.ipynb`, `tidal_heating_correction_test.ipynb` and
`TaylorF2_charged_test.ipynb`.

## Fig. 1: mismatch vs charge

```bash
python scripts/make_fig1.py
```

The script computes the mismatch between the charged and the uncharged (q_BH = 0) waveforms
for the binary configurations of Fig. 1 of arXiv:2402.15336, but at the total mass of the
inference runs, M = 32 Msun, with f_low = 20 Hz and the O5c PSD (match
maximized over t_c and phi_c up to 2 f_ISCO), and saves the figure to
`img/fig1_mismatch_vs_charge.png`. Two alternative sets of configurations (fixed
mass ratio, fixed spins) are commented out at the top of the script.

## Running the inference

All the settings (injected masses and spins, SNRs, priors, sampler options) are in
`config/params.py`; each script has a few switches at the top (`# Edit here`), e.g. the
injected charge `Q_BH_INJECTED`, `FREE_MASSES_AND_SPINS` (5D vs 1D runs),
`UNCHARGED_MODEL` (recovery with $q^{\rm BH}$ fixed to 0) and `TEST_RUN`.
Set `npool` in `SAMPLER_KWARGS` to the number of cores you can use.

### Zero noise, O5c sensitivity (Figs. 2-5)

```bash
python bilby_inference/scripts/make_fig2.py --zero-noise
python bilby_inference/scripts/make_fig3.py --zero-noise
python bilby_inference/scripts/make_fig4.py --zero-noise
python bilby_inference/scripts/make_fig5.py --zero-noise
```

Results and the $q^{\rm BH}$ posterior figure go to `bilby_inference/zero_noise/outdir_figN/`.
Runs are resumed automatically if interrupted.

### Real O4b noise

1. Download the O4b strain from GWOSC (4 kHz, GWF format, H1 and L1) into one folder with the
   subfolders `H1/` and `L1/`, and set `O4B_DATA_DIR` in `config/params.py` to that folder
   (default: `data/O4b` in the repository root). Alternatively, leave `config/params.py`
   unchanged and set the environment variable:
   ```bash
   export O4B_DATA_DIR=/path/to/public-O4b
   ```
2. Build the segment lists and draw the injection times (both detectors in science mode,
   CBC_CAT1, no hardware injections, catalog events vetoed; seed 1234):
   ```bash
   python bilby_inference/scripts/make_segments.py
   ```
   This writes `bilby_inference/segments/injection_times.json`.
3. Run a figure in the n-th noise realization:
   ```bash
   python bilby_inference/scripts/make_fig2.py --index 1
   ```
   Output: `bilby_inference/with_noise/outdir_fig2_O4b_01/`.

`check_real_noise.ipynb` shows the measured ASD and the whitened strain of a realization.

### Einstein Telescope

Download the ET sensitivity curves into `ET_sensitivity/` (see the README there), then

```bash
python bilby_inference/scripts/make_fig_ET.py
```

### Post-processing

```bash
python bilby_inference/scripts/diagnostic_corner.py bilby_inference/zero_noise/outdir_fig2
python bilby_inference/scripts/shared_corner_plots.py --dir-a zero_noise/outdir_fig2 \
       --dir-b with_noise/outdir_fig2_O4b_01 --label-a "zero noise" --label-b "O4b"
```

## Conventions

- Waveform: $\tilde h(f) = A(f)\, e^{-i[\psi(f) + \delta\psi(f)]}$ in the stationary phase
  approximation; phase up to 3.5PN (point particle, including the cubic-in-spin terms), amplitude
  at leading (Newtonian) order.
- $\delta\psi$ contains the 2.5PN, 3.5PN and 4PN tidal heating terms of arXiv:2402.15336, with the
  charge entering as $q^{\rm BH}(M/m_i)^2$ for each black hole. Where a black hole has no horizon,
  $1 - \chi_i^2 + q^{\rm BH}(M/m_i)^2 < 0$, $\delta\psi$ becomes complex.
- Both $\psi$ and $\delta\psi$ are referenced to $f_{\rm ref}$ = 20 Hz (the phase at $f_{\rm ref}$
  is subtracted).
- The waveform is cut at $2 f_{\rm ISCO}$ of the remnant (remnant mass and spin from the NR fits
  in pySEOBNR) with a Tukey taper at the band edges.
- Spins are aligned with the orbital angular momentum (sign of $\chi_i$).
- Posteriors are summarized by the median and the 68% equal-tailed credible interval.

## Validation

- The point-particle TaylorF2 agrees with the `TaylorF2` approximant of LALSimulation for
  non-spinning binaries (phase difference below $10^{-5}$ rad after removing $t_c$ and $\phi_c$);
  with spins they differ only by the cubic-in-spin 3.5PN terms, which LALSimulation does not
  include.
- In the GR limit ($q^{\rm BH} = 0$), $\delta\psi$ agrees to machine precision with the tidal
  heating phase of the code of [arXiv:2311.17554](https://arxiv.org/abs/2311.17554)
  ([gwfit](https://git.ligo.org/samanwaya.mukherjee/gwfit), LIGO GitLab; not distributed here).

## References

- N. V. Krishnendu and S. Chakraborty, *Probing black hole 'charge' from the binary black hole
  inspiral*, Phys. Rev. D 109, 124047 (2024), [arXiv:2402.15336](https://arxiv.org/abs/2402.15336)
- S. Mukherjee, S. Datta, K. S. Phukon and S. Bose, Phenomenological gravitational waveform model of binary black holes
  incorporating horizon fluxes [arXiv:2311.17554](https://arxiv.org/abs/2311.17554),
  Appendix A - TaylorF2 phase coefficients used for the point-particle baseline.
- LIGO O5 design sensitivity: [LIGO-T2500310](https://dcc.ligo.org/LIGO-T2500310).
- Einstein Telescope sensitivity curves: https://gitlab.et-gw.eu/pub/et-sensitivity-curves;
  2L configuration: [arXiv:2303.15923](https://arxiv.org/abs/2303.15923).

## How to cite

If you use this code, please cite it as:

> G. Guarnera, *Code for the thesis 'Constraining the braneworld charge of black holes using
> gravitational waves'*, Zenodo (2026), [doi:10.5281/zenodo.23925540](https://doi.org/10.5281/zenodo.23925540).

The citation metadata are also in `CITATION.cff` ("Cite this repository" on GitHub).

## Acknowledgments

This work uses data obtained from the Gravitational Wave Open Science Center
([gwosc.org](https://gwosc.org)), a service of the LIGO Scientific Collaboration, the Virgo
Collaboration and KAGRA. The inference runs were performed on the BETIF cluster.

## Use of AI tools

An AI assistant (Claude, by Anthropic) was used throughout the development of this code:

- to help implement the code and to debug it;
- to review and clean up the code for publication (consistency between files, docstrings and
  comments, removal of unused code and machine-specific paths);
- to write this README and the documentation in the notebooks.

The modelling choices, the settings of the analyses and the interpretation of the results are
the author's. Every change suggested by the assistant was reviewed by the author, and the code
was checked against independent implementations (see [Validation](#validation)). Any remaining
error is the author's responsibility.

## License

Released under the [MIT License](LICENSE).
