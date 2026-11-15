#!/usr/bin/env bash
# Finish the τ² Telecom test runs, one task at a time, with fresh model servers before each arm.
# A run stops itself when a server stops answering; this restarts the servers and continues from the saved tasks.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STUDENT=${STUDENT:-qwen3-8b}
ARMS=${ARMS:-"program vanilla"}

start_servers() {
  pkill -f "plvr.serve" || true
  sleep 3
  nohup "$PY" -m plvr.models serve "$STUDENT" > "logs/serve-$STUDENT.log" 2>&1 &
  nohup "$PY" -m plvr.models serve qwen3-30b-a3b-instruct > logs/serve-qwen3-30b.log 2>&1 &
  for port in 8108 8130; do
    curl -s --retry 60 --retry-connrefused --retry-delay 3 -o /dev/null "http://127.0.0.1:$port/v1/models"
  done
}

for arm in $ARMS; do
  log="logs/test-$arm-$STUDENT.log"
  for attempt in 1 2 3 4; do
    start_servers
    echo "== $arm attempt $attempt started $(date +%H:%M)" >> "$log"
    if "$PY" -m plvr.run --arm "$arm" --student "$STUDENT" --task-set test --concurrency 1 \
        --out "runs/test/$arm-$STUDENT" >> "$log" 2>&1; then
      break
    fi
  done
  echo "$arm arm finished $(date +%H:%M)" >> "$log"
done
pkill -f "plvr.serve" || true
echo "all arms finished $(date +%H:%M)" >> "logs/test-vanilla-$STUDENT.log"
