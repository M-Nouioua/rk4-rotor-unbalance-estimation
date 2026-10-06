# Raw acquisitions — Bently Nevada RK-4 rotor-kit unbalance campaign

Raw proximity-probe and keyphasor recordings for a two-plane rotor-balancing
campaign, with the condition design and acquisition log needed to interpret
them. The analysis code that consumes these files is archived separately; see
*Companion code* below.

## What is here

| Archive | Contents | Files | Size |
|---|---|---|---|
| `A_severity.tar` | severity ladder, one disk loaded | 192 | 0.73 GB |
| `B_phase.tar` | angular coverage, both disks | 288 | 1.10 GB |
| `C_fine_phase.tar` | fine angular steps | 24 | 0.09 GB |
| `baseline.tar` | balanced baseline and trial-mass calibration | 6 | 0.02 GB |
| `blind.tar` | withheld conditions, labels sealed until after prediction | 51 | 0.19 GB |
| `reference.tar` | run-up sweep and slow-roll runout reference | 2 | 0.05 GB |
| `conditions.csv` | the condition design: masses, angles, radius, block | — | — |
| `index.csv` | acquisition log, one row per record | 561 rows | — |
| `speeds.csv` | dwell-speed definitions | — | — |
| `runup_1x.csv` | order-tracked first-order amplitude and phase against speed | — | — |

Large blocks may be split into parts, named for example
`B_phase.part01.tar`, each a standalone tar that extracts on its own; the table
above gives the per-block totals regardless of how many parts a block occupies.

Totals: 563 recordings, about 2.2 GB. The archives are uncompressed `tar`,
because `.npz` is already a compressed container and re-compressing gains under
5 per cent.

## Record format

Each recording is a NumPy `.npz` archive:

| Key | Type | Meaning |
|---|---|---|
| `fs` | int | sample rate, 12800 Hz per channel |
| `condition_id` | str | key into `conditions.csv` |
| `speed_id` | str | key into `speeds.csv` (`N1`, `N2`) |
| `rpm_commanded` | float | controller setpoint |
| `rpm_measured` | float | speed from the keyphasor pulse train |
| `mount_idx` | int | which physical mounting of the added mass |
| `acq_idx` | int | repeat index within one mounting, 1 to 3 |
| `keyphasor` | float64[102400] | once-per-revolution reference pulse |
| `plane1_x`, `plane1_y` | float64[102400] | probe pair at measurement plane 1 |
| `plane2_x`, `plane2_y` | float64[102400] | probe pair at measurement plane 2 |

Each record is 8.0 s on five channels, so 102400 samples per channel.

**Units.** All five channels are stored as raw **volts** at the Proximitor
output. The probes are calibrated at 7.874 mV/micrometre, equivalent to
200 mV/mil, so

    1 V = 127.0 micrometres = 5.000 mils

The channels are AC-coupled and IEPE excitation is disabled, so the DC gap
voltage, and with it the shaft centreline, is not recorded. Dynamic displacement
and therefore harmonic content, orbits and spectra are preserved; absolute shaft
position is not. Misalignment is consequently not observable in these data.

**Keyphasor.** The pulse is negative-going, reaching about -2.1 V after AC
coupling. Pulses were detected on the falling edge at a threshold of -1.0 V.

## File naming

    {condition_id}_{speed_id}_m{mount_idx}_a{acq_idx}_{unix_timestamp}.npz

For example `BLND04_N1_m1_a2_1784830749.npz` is the second repeat of the first
mounting of condition `BLND04`, acquired at dwell speed `N1`.

## Rig

Bently Nevada RK-4 two-disk rotor kit. Shaft 10 mm diameter, 560 mm between
bearing centres, carrying two 0.800 kg balancing disks of 76 mm diameter and
25 mm thickness. Each disk is drilled with 16 tapped holes on a 30.0 mm pitch
radius at 22.5 degree spacing; eight are engraved at 45 degree intervals. An
added mass of *m* grams at that radius gives an applied unbalance of 30.0*m*
g mm.

Four Bently Nevada 3300 XL proximity probes in orthogonal pairs at two axial
planes, one before the first disk and one after the second, read through the
RK-4 Proximitor assembly. Acquisition on a National Instruments cDAQ-9174 with
two NI-9234 modules, 24-bit, plus or minus 5 V input range.

First critical speed 1648 rpm with damping ratio 0.04 and quality factor 12.4,
and a smaller resonance near 1750 rpm indicating a split critical speed. Both
dwell speeds are sub-critical, measured at 1005 and 1206 rpm, which is 0.61 and
0.73 of the critical speed.

## Reading a record

```python
import numpy as np

d = np.load("A001_N1_m1_a1_1784813244.npz")
fs = int(d["fs"])                      # 12800 Hz
x = d["plane1_x"] * 127.0              # volts -> micrometres
kp = d["keyphasor"]                    # falling edge at -1.0 V marks a revolution
print(float(d["rpm_measured"]), str(d["speed_id"]))
```

Two cautions when deriving first-order amplitude and phase.

**Resample into the angular domain.** Shaft speed varies slightly within a
record, and a fixed-frequency transform leaks. The analysis code resamples to
256 samples per revolution using the measured pulse times, so harmonic vectors
are defined against the keyphasor phase rather than an assumed constant speed.

**Subtract slow-roll runout.** At these sub-critical speeds, mechanical and
electrical runout accounts for 60 to 70 per cent of the raw first-order
amplitude. `reference.tar` contains the slow-roll record taken at 251 rpm for
this purpose. Omitting the correction inflates estimated unbalance by a factor
of 5 to 10 and makes the two dwell speeds disagree.

## Blind conditions

`blind.tar` holds ten conditions whose labels were sealed until predictions from
the influence coefficient method and the two tree ensembles had been recorded.
Their truth values, the seal date, and the file hashes are in the companion
code under `analysis/blind_freeze.json`.

One reference angle in that set, disk 1 of `BLND04`, was logged as 157.0
degrees. The 16-position, 22.5 degree hole circle cannot produce that angle; the
intended position was 157.5 and `conditions.csv` here carries the corrected
value. The frozen record in the companion code retains the sealed value beside
an amendment giving the reason, so the correction is auditable. No recording,
file hash or applied mass was altered.

## Companion code

The processing chain, the four estimators, the evaluation protocol and the audit
scripts are at

- <https://github.com/M-Nouioua/rk4-rotor-unbalance-estimation>
- <https://doi.org/10.5281/zenodo.23152815>

To rebuild the processed feature table from these recordings, place the extracted
`.npz` files and the CSVs in `datasets/` of that repository and run

    python -m scripts.build_features

Every reported result then follows from the pipeline documented in its README.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0).
