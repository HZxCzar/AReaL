#!/usr/bin/env bash
# Source-only helper. Arguments must be child PIDs started with setsid by this
# evaluator, never a GPU-wide/process-name match or the launcher's own group.
eval_stop_process_groups() {
  local pid deadline pending
  local -a owned_pids=()
  for pid in "$@"; do
    [[ "$pid" =~ ^[0-9]+$ ]] && (( pid > 1 )) || continue
    owned_pids+=("$pid")
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  # One shared grace period, not 30 seconds for each individual server.
  deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    pending=0
    for pid in "${owned_pids[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null || kill -0 "$pid" 2>/dev/null; then
        pending=1
        break
      fi
    done
    (( pending )) || break
    sleep 1
  done
  for pid in "${owned_pids[@]}"; do
    if kill -0 -- "-$pid" 2>/dev/null; then
      printf '[cleanup] process group %s did not exit within 30s; sending SIGKILL\n' "$pid" >&2
      kill -KILL -- "-$pid" 2>/dev/null || true
    elif kill -0 "$pid" 2>/dev/null; then
      printf '[cleanup] process %s did not exit within 30s; sending SIGKILL\n' "$pid" >&2
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${owned_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
