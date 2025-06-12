#!/bin/bash
set -e

rm -rf "homegrown_images/$1/"

python3 instrument.py --kind=proc --step="${2:-1}" --no-generate "experiment/$1/main.py" $(head -1 experiment/$1/args.txt)

dirs=($(ls -d homegrown_images/"$1"/*/ | sort -V | grep -v "/0/"))

# Process in order
for dir in "${dirs[@]}"; do
   scripts/homegrown_diff "${dir%/}" &
done

# Wait for all background processes to finish
wait

mkdir -p "results/$1"

# Find next incremental number
i=0
while [ -f "results/$1/homegrown_$i.txt" ]; do
    i=$((i+1))
done

echo "Creating homegrown_$i.txt"

# Concatenate diffsize.txt files in numerical order
for dir in "${dirs[@]}"; do
   cat "${dir%/}/diffsize.txt"
   rm "${dir%/}/"*.bin
done > "results/$1/homegrown_$i.txt"
