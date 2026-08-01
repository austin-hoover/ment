#!/bin/bash
set -euo pipefail

cd ../examples

run() {
  local directory="$1"
  shift

  echo "============================================================"
  echo "Directory: ${directory}"
  echo "============================================================"

  cd "${directory}"

  for script in "$@"; do
    echo "Running: ${directory}/${script}"
    python "${script}"
  done

  cd - >/dev/null
}

run "." \
  fit_cov_2d.py \
  fit_cov_4d.py \
  fit_cov_nd.py \
  rec_2d.py \
  rec_2d_norm.py \
  rec_2d_samp.py \
  rec_4d_2d_rand.py \
  rec_nd_1d_marg.py \
  rec_nd_1d_rand.py \
  rec_nd_2d_marg.py

run "ct" train.py
run "hdr" train.py
run "longitudinal" train.py
run "nonlinear_ring_4d" train.py
run "sampling" test_gm.py test_nurs.py test_samp_2d.py
run "tests" test_diag.py test_interp.py
