#!/bin/bash
set -e

python3 instrument.py --kind=proc --step="${2:-1}" --no-generate "experiment/$1/main.py" $(head -1 experiment/$1/args.txt)
