# Einstein Telescope sensitivity curves

Download from https://gitlab.et-gw.eu/pub/et-sensitivity-curves the two `total.txt` files of one
L-shaped ET detector (cryogenic xylophone design), one for the low-frequency (LF) and one for the
high-frequency (HF) interferometer, and place them here renamed as:

- `totalLF.txt` - `total.txt` of the LF interferometer
- `totalHF.txt` - `total.txt` of the HF interferometer

They are only needed by `bilby_inference/scripts/make_fig_ET.py` (see `get_psd_ET` in `psd.py`).