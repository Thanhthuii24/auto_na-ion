#!/bin/bash
# Usage: tools/gpu_run.sh <EXPERIMENT_ID> <PROPOSAL_JSON>
# Runs ./run_step.sh (preflight + train.py) DETACHED from any agent session,
# so the run survives when the agent exits. Waits until no other train.py is
# running (program.md: only one training run may be active).
# Queue log: experiments/_queue.log
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ID="${1:?experiment id}"; PROP="${2:?proposal json}"
nohup bash -c "
  cd '$ROOT'
  echo \"[\$(date '+%F %T')] QUEUED  $ID $PROP\" >> experiments/_queue.log
  while pgrep -f '^/opt/homebrew/Caskroom/miniforge/base/envs/battery/bin/python train.py --experiment-id' >/dev/null; do sleep 30; done
  echo \"[\$(date '+%F %T')] START   $ID\" >> experiments/_queue.log
  ./run_step.sh '$ID' '$PROP' >> experiments/_queue.log 2>&1
  echo \"[\$(date '+%F %T')] EXIT=\$? $ID\" >> experiments/_queue.log
" >/dev/null 2>&1 < /dev/null &
disown
echo "queued $ID (see experiments/_queue.log; live progress: tail -f live.log)"
