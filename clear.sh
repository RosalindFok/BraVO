#!/bin/bash

handle_error() {
    echo ""
}

set -e
trap handle_error ERR

rm -rf log_* || true

rm state_.log || true

rm slurm-*.out || true

rm -rf __pycache__ || true

set +e
trap - ERR