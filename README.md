# Rotor unbalance estimation on a Bently Nevada RK-4 rotor kit

Code and result artefacts for a study of how the evaluation protocol and the
choice of physical structure affect unbalance estimation from proximity-probe
measurements. Four estimators are compared on the same records: the influence
coefficient method, an extremely randomised trees ensemble, the same ensemble
augmented with the influence coefficient estimate, and a differentiable
physics-informed influence operator.

The rig is a Bently Nevada RK-4 two-plane rotor kit with four 3300 XL proximity
probes, read through the RK-4 Proximitor assembly into an NI cDAQ-9174 chassis
with two NI-9234 modules at 12.8 kS/s per channel.

## Scope

Unbalance only. The proximity channels are AC-coupled, which removes the DC gap
voltage and with it the shaft centreline, so misalignment is not observable
through this front end and is outside the scope of the work.

## What the campaign contains

| Quantity | Value |
|---|---|
| Labelled physical conditions | 85 |
| Labelled acquisitions | 510, three consecutive records at each of two speeds |
| Withheld conditions, labels revealed after prediction | 10, giving 51 records |
| Dwell speeds | 1005 and 1206 rpm, both sub-critical |
| First critical speed | 1648 rpm, with a split near 1750 rpm |
| Applied unbalance | 3 to 60 g mm per disk at a 30.0 mm hole radius |

A condition is one physical mounting of a specified mass and angle. All reported
metrics treat the condition, not the individual record, as the independent unit.

## Data

The raw acquisitions are not in this repository. They are 2.3 GB of `.npz`
records and are archived separately on Zenodo, which is the persistent
identifier named in the paper's data-availability statement.

What is here is everything downstream of them:

- `analysis/features.csv`, the processed feature table the estimators consume
- `analysis/*.json`, the generated result artefacts every reported number comes from
- `analysis/RESULTS.md`, generated from `analysis/benchmark.json`
- `analysis/fig_data/`, the CSV inputs the figure scripts read

With the Zenodo archive in place, `scripts/build_features.py` regenerates
`features.csv` from the raw records, and every result follows from that.

## Layout

| Path | Contents |
|---|---|
| `acquisition/` | NI-DAQmx capture |
| `processing/` | keyphasor order tracking, runout compensation, feature extraction |
| `estimation/` | the four estimators, and the shared statistics in `stats.py` |
| `model/` | rotor model support |
| `scripts/` | the analysis, benchmarking, and audit entry points |
| `matlab/` | figure scripts, Times New Roman, no titles, vector PDF output |
| `analysis/` | generated artefacts and figure data |
| `viz3d/` | a three.js view of the twin state, a side tool |
| `config/`, `config.py` | rig, probe, and acquisition settings |

## Reproducing the results

Install the dependencies, then run the pipeline in this order. Each step writes
a JSON artefact that the next step and the audits read.

    pip install -r requirements.txt

    python -m scripts.build_features        # raw records  -> analysis/features.csv
    python -m scripts.benchmark             # the four estimators, six partitions
    python -m scripts.representation_test   # laboratory frame against rotation-aligned
    python -m scripts.angle_novelty_control # size-matched angular control
    python -m scripts.pinn_gate             # operator variants
    python -m scripts.pinn_modes            # accuracy against mode count
    python -m scripts.pinn_validate         # nested selection and forward validation
    python -m scripts.repeatability         # measurement repeatability

Figures are rendered separately, from the exported CSVs:

    python -m scripts.export_fig_data
    matlab -batch "addpath('matlab'); make_all_figures"

## Checks

Two audits guard against drift between the code and anything quoted from it.

    python -m scripts.audit_numbers   # every headline number against its artefact
    python -m scripts.audit_style     # writing and caption conventions

`scripts/check_sync.py` compares two working copies of this tree by content
hash, and flags any artefact older than the code that generates it. It exists
because an earlier version of that check used an invalid `diff` option, so it
reported success no matter what the trees contained.

## Notes on the analysis

Three points are worth knowing before reading the artefacts.

**The independent unit matters.** Treating all 510 records as independent
replaces 85 physical units with six correlated observations each. Scores
computed per record are retained in the artefacts for continuity with earlier
analyses of this rig, but they describe leakage between repeats rather than
generalisation.

**Two coefficients of determination are reported.** The predictive score is
`1 - SSE/SST`. The fitted-line score is the squared Pearson correlation, which
is insensitive to a calibration slope away from unity, so a shrunk estimator
scores well on it and poorly on the predictive one. The fitted-line score is
labelled a linearity score and is not a measure of accuracy.

**The operator is fitted with restarts.** A single initialisation produced a
four-mode fit that stalled in a local minimum at a training objective of 0.55
against 0.073 from a different start. Any mode-count sweep of a gradient-fitted
physical operator needs restart discipline before a structural conclusion is
drawn from it.

## Licence

MIT, see [LICENSE](LICENSE). This covers the code and the generated artefacts in
this repository. The raw acquisitions are not held here; see Data below.

## Citing

Archived at Zenodo: [10.5281/zenodo.23152815](https://doi.org/10.5281/zenodo.23152815)

## Author

Mourad Nouioua, Interdisciplinary Research Center for Intelligent Manufacturing
and Robotics, King Fahd University of Petroleum and Minerals, Dhahran 31261,
Saudi Arabia.
