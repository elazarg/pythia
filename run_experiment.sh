#!/usr/bin/env bash

EXPERIMENT=$1
if [ -z "$EXPERIMENT" ]; then
  echo "Usage: $0 EXPERIMENT [QMP_PORT] [TCP_PORT]"
  exit 1
fi
shift 1

QMP_PORT=${1:-4444}
TCP_PORT=${2:-1234}

python save_snapshot.py --qmp_port=$QMP_PORT server --tcp_port=$TCP_PORT &

scripts/vm.sh $EXPERIMENT $QMP_PORT $TCP_PORT
