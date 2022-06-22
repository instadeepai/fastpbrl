SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
DOCKER_RUN_FLAGS = --rm --gpus all --ipc="host"
DOCKER_IMAGE_NAME = instadeep/fastpbrl:$(USER)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

.PHONY: build
build:
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f dev.Dockerfile  . --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

.PHONY: exp
exp: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) $(COMMAND)

.PHONY: run_timing_sactd3
run_timing_sactd3:
	/bin/bash timing_scripts/time_train_step.sh

.PHONY: run_timing_dqn
run_timing_dqn:
	/bin/bash timing_scripts/time_train_step_dqn.sh

.PHONY: run_td3
run_td3: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) python3 scripts/train_distributed.py

.PHONY: run_sac
run_sac: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) python3 scripts/train_distributed.py \
		--use-sac

.PHONY: run_td3_cemrl
run_td3_cemrl: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) python3 scripts/train_td3_cemrl.py

.PHONY: run_td3_dvd
run_td3_dvd: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) python3 scripts/train_td3_dvd.py

.PHONY: run_td3_pbt
run_td3_pbt: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) python3 scripts/train_pbt.py

.PHONY: run_sac_pbt
run_sac_pbt: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) python3 scripts/train_pbt.py \
		--use-sac

.PHONY: test_training_scripts
test_training_scripts: build
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) python3 scripts/test_training_scripts.py

.PHONY: dev_container
dev_container: build
	sudo docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_VARS_TO_PASS) -v $(WORK_DIR):/app/fastpbrl $(DOCKER_IMAGE_NAME) /bin/bash
