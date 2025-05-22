#!/bin/bash
# see https://serverfault.com/a/408929/298972

folder="dumps/$1"
mkdir -p "$folder"

grep rw-p "/proc/$1/maps" \
| sed -n 's/^\([0-9a-f]*\)-\([0-9a-f]*\) .*$/\1 \2/p' \
| while read -r start stop; do \
    gdb --batch --pid "$1" -ex \
        "dump memory $folder/$start-$stop.dump 0x$start 0x$stop"; \
done
