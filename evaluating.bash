#!/bin/bash

# ================== Color definitions ==================
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m' # No Color

# ================== tmux session name ==================
SESSION_NAME="evaluating_session"

# ================== Project paths ==================
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/car_ros/scripts/racing_comp_blocking.py"

# ================== Kill existing session ==================
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Session '$SESSION_NAME' already exists. Killing it...${NC}"
    tmux kill-session -t "$SESSION_NAME"
fi

# ================== Start tmux session ==================
echo -e "${GREEN}Starting tmux session '$SESSION_NAME'...${NC}"

# opp1: mpc ibr rl ours-low_data ours-low_p
# opp2: mpc ibr rl ours-low_data ours-low_p
tmux new-session -d -s "$SESSION_NAME" "
    cd \"$PROJECT_ROOT\" && \
    python3 \"$SCRIPT_PATH\" \
        --opp1 mpc \
        --opp2 mpc \
        --mpc \
        --rel
    EXIT_CODE=\$?
    echo
    echo -e \"${GREEN}Training finished with exit code \$EXIT_CODE${NC}\"
    sleep 1
    tmux kill-session -t \"$SESSION_NAME\"
"

# ================== Attach and wait ==================
echo -e "${GREEN}Attaching to tmux session '$SESSION_NAME'...${NC}"
tmux attach -t "$SESSION_NAME"

# ================== Back to shell ==================
echo -e "${GREEN}tmux session '$SESSION_NAME' closed. Back to shell.${NC}"


