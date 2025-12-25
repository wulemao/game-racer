#!/bin/bash
SESSION_NAME="multi_car"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Stopping tmux session $SESSION_NAME"
  tmux kill-session -t "$SESSION_NAME"
else
  echo "Session $SESSION_NAME not found"
fi