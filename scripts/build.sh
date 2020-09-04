#!/bin/bash
source activate tf2
book=site/
#get most recent notebook.
jupyter nbconvert --to script convert.ipynb
python convert.py
jupyter-book build ../$book
