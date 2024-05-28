#!/bin/bash

# Find the PID of the process using port 8080
pid=$(lsof -t -i :8080)

# Check if the PID is not empty
if [ -n "$pid" ]; then
  echo "Stopping Streamlit application running on port 8080 with PID: $pid"
  kill $pid

  # Optionally, you can wait for a few seconds and check if the process is still running
  sleep 5
  if kill -0 $pid > /dev/null 2>&1; then
    echo "Process $pid did not stop, forcefully killing it..."
    kill -9 $pid
  else
    echo "Process $pid stopped successfully."
  fi
else
  echo "No process is running on port 8080."
fi
