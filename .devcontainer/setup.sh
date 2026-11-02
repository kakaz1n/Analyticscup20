#!/usr/bin/env bash
# First start of a Codespace: download the SkillCorner open data from its official sources and build everything
# (pose features, moments, 3D skeletons, simulations, error patterns, validation). ~45 min on 4 cores, once.
set -euo pipefail
if [ -f data/build/validation.json ] && [ -d data/build/1996435/full ]; then
  echo "Data already built."; exit 0
fi
echo ">>> Building Defensive Body Language (one-off, ~45 min). The viewer opens on port 8765 when it is done."
python -m dbl all 2>&1 | tee data-build.log
