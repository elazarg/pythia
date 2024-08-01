#!/usr/bin/env bash

EXPERIMENT=$1
if [ -z "$EXPERIMENT" ]; then
  echo "Usage: $0 EXPERIMENT [QMP_PORT] [TCP_PORT]"
  exit 1
fi

STEP=${2:-1}
QMP_PORT=${3:-4444}
TCP_PORT=${4:-1234}

shift 4

set -x

python save_snapshot.py --qmp_port=$QMP_PORT server --tcp_port=$TCP_PORT &

scripts/vm.sh $EXPERIMENT $STEP $QMP_PORT $TCP_PORT
