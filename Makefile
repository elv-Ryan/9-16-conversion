IMAGE_NAME := verticalvideo-v2
PYTHON ?= python3

include buildscripts/Makefile.tagger-model

.PHONY: unit-test podman-test protocol-test

NBA_YOLO_TESTS := \
	tests/test_common_ml_contract.py \
	tests/test_config.py \
	tests/test_family_routing.py \
	tests/test_joe_shot_filename_parser.py \
	tests/test_manifest.py \
	tests/test_output_validator.py \
	tests/test_trajectory.py

unit-test:
	@set -e; \
	for test_file in $(NBA_YOLO_TESTS); do \
		echo "===== $$test_file ====="; \
		PYTHONPATH=src $(PYTHON) $$test_file; \
	done

podman-test:
	@test -n "$(TEST_SHOT)" || (echo "Set TEST_SHOT=/absolute/path/to/shot.mp4" && exit 2)
	DEVICE=$${DEVICE:-0} IMAGE=$(IMAGE_NAME):$(IMAGE_TAG) ./scripts/test_podman_shot.sh "$(TEST_SHOT)"

protocol-test: podman-test
