#!/usr/bin/env bash
set -euo pipefail

data_path="${1:-data/processed/model_input.csv}"
output_dir="${2:-outputs/private-v4c}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v snd-control >/dev/null 2>&1; then
  cli=(snd-control)
else
  export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
  cli=(python -m wastewater_snd)
fi

if [[ ! -f "${data_path}" ]]; then
  printf 'Data file not found: %s\n' "${data_path}" >&2
  printf 'Place a private contract-compliant CSV under data/processed/, or pass its path as the first argument.\n' >&2
  exit 2
fi

"${cli[@]}" validate --data "${data_path}"
"${cli[@]}" train-calibrated \
  --data "${data_path}" \
  --output-dir "${output_dir}" \
  --required-r2 0.8

printf 'Training completed. Evaluation: %s/calibrated_evaluation.csv\n' "${output_dir}"
printf 'Model bundle: %s/calibrated_model_bundle.joblib\n' "${output_dir}"
printf 'Use synthetic config examples only as schemas; replace them locally with authorized same-batch measurements.\n'
