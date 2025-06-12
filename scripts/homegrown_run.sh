#!/bin/bash
set -e

python3 instrument.py --kind=proc --step="${2:-1}" --no-generate "experiment/$1/main.py" $(head -1 experiment/$1/args.txt)

for f in homegrown_images/"$1"/*/parent; do scripts/homegrown_diff "$(dirname "$f")"; done

# Wait for all background processes to finish
wait

mkdir -p "results/$1"
cat homegrown_images/"$1"/*/diffsize.txt > "results/$1/homegrown.txt"

# Remove the temporary directory
rm -rf "homegrown_images/$1/"
