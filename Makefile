export SHELL := /bin/bash

.PHONY: build clean

build:
	python setup.py build_ext --inplace 

clean:
	find . -type f -name '*.so' -delete
	find . -type f -name '*.cpp' -delete
	rm -rf build/

test:
	python -m unittest discover .

