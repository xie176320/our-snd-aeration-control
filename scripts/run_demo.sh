#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-outputs/demo}"
data_path="${output_dir}/demo_model_input.csv"
model_dir="${output_dir}/model"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${output_dir}"

if command -v snd-control >/dev/null 2>&1; then
  cli=(snd-control)
else
  export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
  cli=(python -m wastewater_snd)
fi

"${cli[@]}" demo-data --output "${data_path}"
"${cli[@]}" validate --data "${data_path}"
"${cli[@]}" train-calibrated \
  --data "${data_path}" \
  --output-dir "${model_dir}" \
  --required-r2 0.8
"${cli[@]}" predict-calibrated \
  --model "${model_dir}/calibrated_model_bundle.joblib" \
  --calibration configs/calibration_three_point.example.csv \
  --condition configs/calibrated_condition.example.json

printf 'Demo completed. Results: %s\n' "${model_dir}"
