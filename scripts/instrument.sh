#!/bin/bash

instrument() {
  python instrument.py --kind=proc --no-generate $(head -1 experiment/$1/flags.txt) experiment/$1/main.py $(head -1 experiment/$1/args.txt)
  ls -lv -s --block-size=64 criu_images/$1/*/pages-1.img
}

instrument $1
