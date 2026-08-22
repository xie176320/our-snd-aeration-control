#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ -n "${PYTHON:-}" ]]; then
  python_cmd="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
else
  printf 'Python 3.10-3.12 was not found. Install Python, then run this script again.\n' >&2
  exit 2
fi

venv_python="${project_root}/.venv/bin/python"
if [[ ! -x "${venv_python}" ]]; then
  printf 'Creating local virtual environment...\n'
  "${python_cmd}" -m venv "${project_root}/.venv"
fi

if ! "${venv_python}" -c 'import streamlit, wastewater_snd' >/dev/null 2>&1; then
  printf 'Installing local dashboard dependencies...\n'
  "${venv_python}" -m pip install --upgrade pip
  "${venv_python}" -m pip install -e ".[web,plot]"
fi

export SND_APP_MODE=local
export SND_LOCAL_IMPORT=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

port="${SND_PORT:-8501}"
printf 'Opening the local-only importer at http://127.0.0.1:%s\n' "${port}"

(
  sleep 2
  if command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:${port}"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:${port}"
  fi
) >/dev/null 2>&1 &

exec "${venv_python}" -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port "${port}" \
  --server.headless true
