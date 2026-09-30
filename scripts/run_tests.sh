#!/usr/bin/env bash
# Run the validation tests of steps 1-4 (headless). Takes ~6 minutes.
# Usage: scripts/run_tests.sh [seeds_for_episode_tests]   (default 8)
set -e
cd "$(dirname "$0")/.."
SEEDS="${1:-8}"
python3 test_step1_kinematics.py
python3 test_step2_sensors.py
python3 test_step3_env.py "$SEEDS"
python3 test_step4_bg.py "$SEEDS"
