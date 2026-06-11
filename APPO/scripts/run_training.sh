#!/bin/bash
# Wrapper script to run APPO training with logging

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/appo_train.log}"

echo "Starting APPO training at $(date)" >> "${LOG_FILE}"

bash "${SCRIPT_DIR}/APPO_7B_Reasoning_1node.sh" >> "${LOG_FILE}" 2>&1

echo "Training finished at $(date), exit code: $?" >> "${LOG_FILE}"
