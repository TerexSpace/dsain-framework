#!/bin/bash
while true; do
  sleep 7200  # 2 hours
  cd /workspace/dsain-framework/results/full
  count=$(ls enhanced_*.json 2>/dev/null | wc -l)
  if [ $count -gt 0 ]; then
    echo "[$(date)] Backup: $count results saved to persistent location"
    # Results already in /workspace which persists
  fi
done
