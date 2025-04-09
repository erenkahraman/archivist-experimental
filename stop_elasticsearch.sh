#!/bin/bash
pid=$(pgrep -f elasticsearch)
if [ -n "$pid" ]; then
  echo "Stopping Elasticsearch (PID: $pid)..."
  kill $pid
  echo "Elasticsearch stopped"
else
  echo "Elasticsearch not running"
fi
