
## Experimental v1.3.0 candidate (not yet archived)

- Revises the synthetic mission generator to include weak/moderate/strong event intensity, partial modality conflict, and limited overlap with elevated normal background conditions.
- Preserves the 160-record mission and the original three event windows.
- Fits the baseline scaler and Isolation Forest models once and reuses them for perturbation testing.
- Robustness at 5%, 10%, and 20% is summarized across 30 independent Gaussian feature-noise realizations rather than a single random seed.
- The revision is intended to avoid trivially separable synthetic labels; it does not target any predetermined accuracy or F1 score.
- Streamlit UI reports Prototype v1.3.0, displays robustness noise levels as 0%, 5%, 10%, and 20%, and summarizes robustness metrics as mean ± SD while retaining full raw CSV outputs.

# Release notes — Version 1.2.0

## Purpose

Version 1.2.0 is the reviewer-resubmission reproducibility release aligned with the revised IJ-AI manuscript **“Advanced Drone Sensing Framework for Smart X Situational Intelligence.”**

## Included reviewer-revision capabilities

- explicit two-branch Isolation Forest configuration;
- 200 estimators per branch and contamination 0.10;
- fixed branch seeds 12 and 24;
- baseline alpha 0.60 and alpha-sensitivity analysis at 0.20, 0.40, 0.50, 0.60, and 0.80;
- explicit Normal/Watch/Warning/Critical threshold ordering (0.45/0.60/0.75);
- quantitative baseline metrics against known synthetic labels;
- scenario-level detection results;
- pollutant-only, image-only, and ensemble modality ablation;
- controlled feature-space noise robustness at 0%, 5%, 10%, and 20%;
- persistent human-validation logging and parameter snapshots;
- standalone experiment runner and paper-ready CSV/JSON outputs;
- locked dependencies and data dictionary.

## Changes relative to the earlier archive

- release metadata and citation metadata are synchronized to Version 1.2.0;
- README language is synchronized with the revised manuscript and explicitly limits claims to controlled synthetic feasibility;
- obsolete GitHub-release instructions and placeholder repository metadata are removed;
- no field-extension, YOLO, Boreal, Gunung Lokon, or collaborative multi-view functionality is included;
- Python cache artifacts are removed from the public release;
- manuscript-to-artifact mapping and release notes are added for easier audit.

## Scientific scope

No new field dataset or field-accuracy claim is introduced in this release. The underlying controlled experiment remains the evidence base for the quantitative manuscript results.

### v1.3.1 candidate: single-drone empirical visual branch
- Adds optional frozen YOLO smoke inference for one uploaded drone image.
- Uses the previously trained Boreal Forest Fire UAV smoke detector when `models/best.pt` is supplied.
- Reports smoke count, maximum/mean confidence, bounding-box area ratio, visual evidence score, and visual evidence level.
- Adds a persistent single-drone inference CSV log.
- Does not include multi-drone selection, multi-view fusion, or a new field-accuracy claim.


### v1.3.2 candidate: corrected YOLO evidence reporting
- Candidate boxes and accepted smoke detections are now reported separately.
- Added an independent accepted-smoke threshold (default 0.45, explicitly provisional).
- Replaced summed detection-area ratio with exact union bounding-box coverage.
- Candidate and accepted box coverage are reported separately.
- Maximum smoke confidence remains the visual score.
- Controlled synthetic Isolation-Forest experiments are unchanged.
