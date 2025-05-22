#!/bin/bash

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (use sudo)" >&2
  exit 1
fi
criu service --shell-job --address /tmp/criu_service.socket
