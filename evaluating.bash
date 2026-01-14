#!/usr/bin/env bash
set -euo pipefail

# ================== Color definitions ==================
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m' # No Color

# ================== tmux session names ==================
SESSION_NAME="evaluating_session"          # training session
VIZ_SESSION_NAME="foxglove_session"        # foxglove_bridge session
ROSCORE_SESSION_NAME="roscore_session"     # roscore session (optional)

# ================== Project paths ==================
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/car_ros/scripts/racing_comp_blocking.py"

# ================== Options ==================
START_ROSCORE=true   # set to false if you do NOT want the script to start roscore

# ================== Cleanup function ==================
cleanup() {
  echo -e "${YELLOW}\n[Cleanup] Killing tmux sessions...${NC}"

  # Kill training session
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME" || true
  fi

  # Kill foxglove_bridge session
  if tmux has-session -t "$VIZ_SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$VIZ_SESSION_NAME" || true
  fi

  # Kill roscore session (optional)
  if tmux has-session -t "$ROSCORE_SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$ROSCORE_SESSION_NAME" || true
  fi

  echo -e "${GREEN}[Cleanup] Done.${NC}"
}

# Ensure cleanup runs on normal exit, Ctrl+C, or termination
trap cleanup EXIT INT TERM

# ================== Kill existing sessions ==================
for s in "$SESSION_NAME" "$VIZ_SESSION_NAME" "$ROSCORE_SESSION_NAME"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo -e "${YELLOW}Session '$s' already exists. Killing it...${NC}"
    tmux kill-session -t "$s"
  fi
done

# ================== Start roscore (optional) ==================
if [ "$START_ROSCORE" = true ]; then
  echo -e "${GREEN}Starting tmux session '$ROSCORE_SESSION_NAME' (roscore)...${NC}"
  tmux new-session -d -s "$ROSCORE_SESSION_NAME" "
    set -e
    echo -e \"${GREEN}[roscore] starting...${NC}\"
    roscore
  "
  # Give roscore some time to initialize
  sleep 1
fi

# ================== Start foxglove_bridge session ==================
echo -e "${GREEN}Starting tmux session '$VIZ_SESSION_NAME' (foxglove_bridge)...${NC}"
tmux new-session -d -s "$VIZ_SESSION_NAME" "
  set -e
  echo -e \"${GREEN}[foxglove_bridge] starting...${NC}\"
  rosrun foxglove_bridge foxglove_bridge
"

# ================== Start evaluating session ==================
tmux new-session -d -s "$SESSION_NAME" "
    cd \"$PROJECT_ROOT\" && \
    python3 \"$SCRIPT_PATH\"
    
    EXIT_CODE=\$?
    echo
    if [ \$EXIT_CODE -ne 0 ]; then
        echo -e \"${RED}Error: Process failed with exit code: \$EXIT_CODE${NC}\"
    else
        echo -e \"${GREEN}Process finished successfully with exit code: \$EXIT_CODE${NC}\"
    fi
    
    echo -e \"${YELLOW}Note: Session kept alive for debugging. Type 'exit' to close window.${NC}\"
    # Transition to an interactive bash shell to keep the window open
    exec bash
"

# ================== Attach and wait ==================
echo -e "${GREEN}Attaching to tmux session '$SESSION_NAME'...${NC}"
tmux attach -t "$SESSION_NAME" || true

# ================== Back to shell ==================
echo -e "${GREEN}tmux session '$SESSION_NAME' closed. Back to shell.${NC}"
