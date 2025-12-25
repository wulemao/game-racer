#!/bin/bash
set -e

SESSION_NAME="multi_car"

ROS1_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="$HOME/ddrx_ws/devel/setup.bash"

PKG_NAME="car_ros"
NODE_SCRIPT="racing_blocking.py"

ENV_SETUP="export XLA_PYTHON_CLIENT_PREALLOCATE=false; export JAX_PLATFORM_NAME=cpu; export XLA_FLAGS='--xla_force_host_platform_device_count=1'; export OMP_NUM_THREADS=1"
ROS_SETUP_CMD="source $ROS1_SETUP; source $WS_SETUP"

echo "Allow about 50 seconds in total for all windows to start"

echo "Creating window data0"
tmux new-session -d -s "$SESSION_NAME" -n "data0" \
"bash -lc '$ENV_SETUP; $ROS_SETUP_CMD; rosrun $PKG_NAME $NODE_SCRIPT --exp_name data0_multi; echo \"[data0 done] Press any key to exit...\"; read -n 1'"

sleep 5

for i in {1..9}
do
  echo "Creating window data$i"
  tmux new-window -t "$SESSION_NAME" -n "data$i" \
  "bash -lc '$ENV_SETUP; $ROS_SETUP_CMD; rosrun $PKG_NAME $NODE_SCRIPT --exp_name data${i}_multi; echo \"[data${i} done] Press any key to exit...\"; read -n 1'"
  sleep 5
done

tmux attach -t "$SESSION_NAME"
