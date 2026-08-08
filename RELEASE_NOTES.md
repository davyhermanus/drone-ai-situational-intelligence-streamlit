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
