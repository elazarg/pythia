#!/bin/bash

python3 "$@" &
PID=$!

while kill -0 $PID 2>/dev/null; do
    if kill -s SIGINT $PID; then
        gcore -o /path/to/dumps/core $PID
        if [ -n "$OLD_CORE_DUMP" ]; then
          diff_coredump "$OLD_CORE_DUMP" "$NEW_CORE_DUMP"
          mv "$NEW_CORE_DUMP" "$OLD_CORE_DUMP"
        else
          OLD_CORE_DUMP="$NEW_CORE_DUMP"
        fi
    fi
    sleep 1
done
