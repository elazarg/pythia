#!/usr/bin/python3.9
import sys
import pickle
import os.path

with open(sys.argv[1], 'rb') as infile:
    while True:
        try:
            x = pickle.load(infile)
            print(x[0], [count for count, diff in x[1]])
        except EOFError:
            break
    
