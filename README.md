# Calibration-aware symbolic regression for interpretable sepsis mortality prediction under hospital shift

Code and reproducibility materials for the manuscript submitted to *Artificial Intelligence in Medicine*:

> Shaheen N, Dang BT. Calibration-aware symbolic regression for interpretable sepsis mortality prediction under hospital shift. (Manuscript under review.)

Genetic-programming symbolic regression (via [PySR](https://github.com/MilesCranmer/PySR)) is used to evolve a compact, auditable sepsis mortality formula from first-24-hour ICU variables, compared against logistic regression, random forest, XGBoost, Platt-calibrated baselines, and APACHE-IV. The model is evaluated for discrimination and calibration across repeated patient/hospital partitions, a nine-cell hospital-mortality tertile transfer matrix, and zero-shot external validation on MIMIC-IV.

This work originated as the author's MSc thesis (*Evolving Interpretable Sepsis Mortality Risk Scores via Genetic Programming*, Whitecliffe College, 2026); this repository contains the subset of code and results relevant to the journal manuscript. Notebooks `M1`–`M4` were added during manuscript revision to address supervisor feedback and are not part of the original thesis pipeline.

## Repository structure

```
notebooks/                     Core pipeline, run in order NB01 -> NB15
  NB01_overview_eda.ipynb           Exploratory data analysis
  NB02_cohort_construction.ipynb    Sepsis-3 cohort construction (eICU-CRD)
  NB03_feature_engineering.ipynb    Feature extraction (first 24h)
  NB04_hospital_stratification.ipynb
  NB05_feature_curation.ipynb       26-variable GP terminal set
  NB06_apache4_baseline.ipynb       APACHE-IV comparator
  NB07_split_materialisation.ipynb  Train/val/test + hospital splits
  NB08_baseline_models.ipynb        LR / RF / XGB (+ Platt calibration)
  NB09_feature_importance.ipynb     XGBoost SHAP (terminal-set curation)
  NB10_gp_symbolic_regression.ipynb Main GP run (PySR, BCE objective)
  NB11_model_comparison.ipynb       Test-set model comparison
  NB11_repro_runs.ipynb             30-run i.i.d. partition reproducibility
  NB12_hospital_out_evaluation.ipynb  52/13 hospital-out evaluation
  NB12_repro_runs.ipynb             30-run hospital-out reproducibility
  NB13_tertile_matrix_cmaes.ipynb   9-cell tertile matrix + CMA-ES recalibration
  NB14_xai_clinical_interpretation.ipynb  Random Forest SHAP (GP feature agreement)
  NB15_mimic_external_validation.ipynb    Zero-shot MIMIC-IV validation

  manuscript/                  Revision analyses (see manuscript, Methods)
    M1_feature_compression_table.ipynb   Feature-selection disposition table
    M2_consensus_selection.ipynb         Cross-run expression consensus ranking
    M3_prevalence_recalibration.ipynb    Prevalence-adaptive recalibration vs. CMA-ES
    M4_reduced_feature_baselines.ipynb   10-feature-matched baseline retrain

  development/                 Pilot analyses exploring alternative GP loss functions
    NB10_pilot_custom_loss.ipynb         BCE vs. sigmoid-MSE pilot comparison
    NB10_pilot_logitdistloss.ipynb       LogitDistLoss pilot (rejected approach)
    NB11_fixed_model_partition_check.ipynb  10-group fixed-model partition check

src/                            Shared utilities (metrics, path config)
configs/paths.yml               Path configuration template — edit before running

results/
  shared/                      NB01-09 outputs (cohort, EDA, baseline tables/figures)
  v2_bce/                      NB10-15 outputs (BCE-loss GP, canonical results)
  manuscript/                  M1-M4 outputs
```

## Data access

This study uses eICU-CRD v2.0 and MIMIC-IV v3.1, both credentialed-access datasets distributed via [PhysioNet](https://physionet.org/). **Raw and patient-level derived data are not included in this repository**, in line with the PhysioNet data use agreement. To reproduce the pipeline from raw data:

1. Obtain credentialed access to [eICU-CRD](https://physionet.org/content/eicu-crd/2.0/) and [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) via PhysioNet.
2. Edit `configs/paths.yml` to point at your local copies.
3. Run `notebooks/NB01`–`NB15` in order; each writes its outputs under `results/`.

All files under `results/` in this repository are aggregate/summary statistics (per-run metrics, SHAP rankings, calibration tables) — no patient-level rows or record identifiers are included, with one exception: `results/v2_bce/tables/NB14_patient_case_studies.csv` contains four illustrative case rows (true/false positive/negative) with the eICU-CRD record identifier removed.

## Environment

```
conda env create -f environment.yml
conda activate sepsis-gp
```

Symbolic regression (NB10, M2) requires PySR's Julia backend; see the [PySR installation guide](https://ai.damtp.cam.ac.uk/pysr/) for setup.

## License

Code is released under the MIT License (see `LICENSE`). This license covers the code in this repository only — it does not extend any rights to the eICU-CRD or MIMIC-IV datasets, which remain governed by their respective PhysioNet data use agreements.

## Citation

A full citation will be added once the manuscript is published. In the meantime, please cite via this repository.
