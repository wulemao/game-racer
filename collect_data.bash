#!/bin/bash
set -euo pipefail

SESSION_NAME="multi_car"

ROS1_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="$HOME/ddrx_ws/devel/setup.bash"

PKG_NAME="car_ros"
NODE_SCRIPT="racing_blocking.py"

FOXGLOVE_PKG="foxglove_bridge"
FOXGLOVE_NODE="foxglove_bridge"
FOXGLOVE_ADDRESS="0.0.0.0" # for local-only, use 127.0.0.1
FOXGLOVE_PORT="8765"

ENV_SETUP="export XLA_PYTHON_CLIENT_PREALLOCATE=false; export JAX_PLATFORM_NAME=cpu; export XLA_FLAGS='--xla_force_host_platform_device_count=1'; export OMP_NUM_THREADS=1"
ROS_SETUP_CMD="source $ROS1_SETUP; source $WS_SETUP"

LOG_DIR="${HOME}/ddrx_ws/src/game-racer/logs/${SESSION_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m'

cleanup_existing_session() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Session '$SESSION_NAME' already exists. Killing it...${NC}"
    tmux kill-session -t "$SESSION_NAME"
  fi
}

mkcmd_node() {
  local name="$1"      # data0, data1, ...
  local exp="$2"       # data0_multi, ...
  local logfile="$3"

  cat <<EOF
bash -lc '
set -euo pipefail
$ENV_SETUP
$ROS_SETUP_CMD

# ================================
# ROS namespace per tmux window
# ================================
export ROS_NAMESPACE=$name

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[\$(ts)] START $name exp_name=$exp namespace=\$ROS_NAMESPACE" | tee -a "$logfile"

set +e
rosrun $PKG_NAME $NODE_SCRIPT --exp_name "$exp" 2>&1 | tee -a "$logfile"
rc=\${PIPESTATUS[0]}
set -e

if [ "\$rc" -ne 0 ]; then
  echo -e "${RED}[\$(ts)] FAIL $name (exit=\$rc). Log: $logfile${NC}" | tee -a "$logfile"
  exit "\$rc"
else
  echo -e "${GREEN}[\$(ts)] OK   $name. Log: $logfile${NC}" | tee -a "$logfile"
fi

echo "[done] Press any key to exit..."
read -n 1
'
EOF
}

mkcmd_roscore() {
  local logfile="$1"
  cat <<EOF
bash -lc '
set -euo pipefail
$ROS_SETUP_CMD

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[\$(ts)] START roscore" | tee -a "$logfile"

set +e
roscore 2>&1 | tee -a "$logfile"
rc=\${PIPESTATUS[0]}
set -e

if [ "\$rc" -ne 0 ]; then
  echo -e "${RED}[\$(ts)] FAIL roscore (exit=\$rc). Log: $logfile${NC}" | tee -a "$logfile"
  exit "\$rc"
fi

echo "[roscore ended] Press any key to exit..."
read -n 1
'
EOF
}

mkcmd_foxglove() {
  local logfile="$1"
  cat <<EOF
bash -lc '
set -euo pipefail
$ROS_SETUP_CMD

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[\$(ts)] START foxglove_bridge address=$FOXGLOVE_ADDRESS port=$FOXGLOVE_PORT" | tee -a "$logfile"

set +e
rosrun $FOXGLOVE_PKG $FOXGLOVE_NODE _address:=$FOXGLOVE_ADDRESS _port:=$FOXGLOVE_PORT 2>&1 | tee -a "$logfile"
rc=\${PIPESTATUS[0]}
set -e

if [ "\$rc" -ne 0 ]; then
  echo -e "${RED}[\$(ts)] FAIL foxglove_bridge (exit=\$rc). Log: $logfile${NC}" | tee -a "$logfile"
  exit "\$rc"
else
  echo -e "${GREEN}[\$(ts)] OK   foxglove_bridge. Log: $logfile${NC}" | tee -a "$logfile"
fi

echo "[foxglove_bridge ended] Press any key to exit..."
read -n 1
'
EOF
}

echo "Logs: $LOG_DIR"
echo "Allow about 50 seconds in total for all windows to start"

cleanup_existing_session

echo "Creating session and window roscore"
tmux new-session -d -s "$SESSION_NAME" -n "roscore" "$(mkcmd_roscore "$LOG_DIR/roscore.log")"

sleep 2

echo "Creating window foxglove"
tmux new-window -t "$SESSION_NAME" -n "foxglove" "$(mkcmd_foxglove "$LOG_DIR/foxglove.log")"

sleep 2

echo "Creating window data0"
tmux new-window -t "$SESSION_NAME" -n "data0" "$(mkcmd_node "data0" "data0_multi" "$LOG_DIR/data0.log")"
sleep 5

for i in {1..9}
do
  echo "Creating window data$i"
  tmux new-window -t "$SESSION_NAME" -n "data$i" "$(mkcmd_node "data$i" "data${i}_multi" "$LOG_DIR/data${i}.log")"
  sleep 5
done

tmux attach -t "$SESSION_NAME"
