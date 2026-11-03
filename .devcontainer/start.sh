#!/usr/bin/env bash
# Every Codespace start: launch the 3D viewer in the background on port 8765 (log in viewer.log).
if [ ! -f data/build/validation.json ]; then
  echo "Data not built yet: run 'bash .devcontainer/setup.sh' (or wait for the first build to finish), then 'bash .devcontainer/start.sh'."
  exit 0
fi
if curl -s -o /dev/null localhost:8765/; then echo "Viewer already running on port 8765."; exit 0; fi
nohup setsid python -m dbl viewer --no-browser --host 0.0.0.0 > viewer.log 2>&1 < /dev/null &
for _ in $(seq 1 60); do
  if curl -s -o /dev/null localhost:8765/; then echo "Viewer running: open the PORTS tab -> 8765."; exit 0; fi
  sleep 1
done
echo "Viewer failed to start:"; tail -20 viewer.log
