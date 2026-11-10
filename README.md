# HW 1 - Language models from scratch

This package contains the experimental evidence for the accompanying report. It uses the nanolm implementation in [DeepDataMiningLearning](https://github.com/lkk688/DeepDataMiningLearning). Credit for the training implementation and tutorial belongs to that project; the analysis here summarizes the saved homework runs.

## Contents

- `runs/`: metrics, training/evaluation logs and original per-run metadata.
- `configs/`: portable resolved configurations for 26 unique runs.
- `original_colab_configs/`: original supplied configurations.
- `analysis/architecture_vs_lstm/`: original analyzer output, including summary.csv and summary.md.
- `attention/`: three input texts, raw attention arrays, original plots and measurements.
- `report_analysis/`: regenerated figures, statistical tables and metric checks.
- `scripts/`: plotting and measurement scripts.
- `run_manifest.json`, `version_check.txt`: provenance and compute accounting.
- `Report_editable.md`: editable report text (HTML formatting within paragraphs).

There are no model checkpoints or training datasets in this ZIP. Keep the existing Colab/Drive checkpoint if generating new samples or attention maps. Extra credit was not performed.

## Reproduce analysis without retraining

Install numpy, pandas, scipy and matplotlib in a Python environment. From this directory:

```bash
python scripts/analyze_results.py
python scripts/measure_attention.py
```

The scripts use paths relative to their own location and regenerate the report_analysis outputs. The attention script restricts word-start measurements to third and later letters to distinguish word start from the immediately previous character. Three-text summary means weight texts equally.

## Reproduce training

Clone the upstream repository and check out commit `8418fc1bae85657190f11749483bf2e729b9702c`. Follow its dependency installation instructions, then work from `DeepDataMiningLearning/llm`. The original architecture runs used `cc64ca3ed3aa1ce7df3cf381c492bc7a9826b749`; the user verified no changes in the nanolm directory between these versions. Numerical results can vary with hardware and dependency versions.

Run a portable configuration using its absolute path, for example:

```bash
python -m nanolm.train --config /absolute/path/HW1_Supporting_Files/configs/architecture/gpt_modern_seed1337.yaml
```

All 26 configuration files are independent. Do not rerun 256-context experiments: use the matching architecture runs. Other contexts use batch size 8192 / context length and 2000 steps. The two untrained runs use one step with zero learning rates and weight decay, preserving initialized weights. Configurations redirect output paths but preserve numerical settings; inspect them before launching a long job.

To reproduce a visualization with an available checkpoint:

```bash
python -m nanolm.viz attention --run /absolute/path/to/run --text "The little bird flew over the garden and into the trees."
```

Repeat for the texts in attention/texts.json. Copy each generated image before the next command because the visualization command can overwrite its default output filename.

## Interpretation and submission

Lower bpb is better. Reported SD uses two seeds, not a confidence interval. Saved paired p-values are unadjusted and exploratory. Training recipes differ in learning rate and weight decay. Context experiments change both training and evaluation length, and validation token coverage varies slightly.

Review the report, add your name/course details if required, and submit the single PDF to Canvas. Upload this extracted folder to Google Drive or GitHub and submit its accessible link separately. The report and this archive have not been published or submitted automatically.
