IMAGE_NAME := verticalvideo
ELUVIO_MAKEFILE_CUSTOM_TEST := 1

include buildscripts/Makefile.tagger-model

test:
	./test.sh

unit-test:
	@if [ -x .venv/bin/python ]; then \
		PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v; \
	else \
		PYTHONPATH=src python3 -m unittest discover -s tests -v; \
	fi

.PHONY: test unit-test
