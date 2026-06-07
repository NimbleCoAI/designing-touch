#!/bin/bash
# Double-click this file (Finder) to launch the designing-touch live preview.
# It sets up the Python environment on first run, then opens the window.

cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
  echo "First run — setting up (this takes a minute)…"
  python3 -m venv .venv || { echo "Could not create venv. Is python3 installed?"; read -r; exit 1; }
  ./.venv/bin/pip install --upgrade pip >/dev/null 2>&1
  ./.venv/bin/pip install -e . || { echo "Install failed — see messages above."; read -r; exit 1; }
fi

echo "Launching… close the window (red X) to quit."
exec ./.venv/bin/python experiments/05-live-webcam/run.py "$@"
