#!/bin/bash
book=introml/
#cd scripts && python convert.py
#jupyter-book toc ./$book
jupyter-book build $book
