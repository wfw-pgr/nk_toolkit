#!/bin/bash

_SETVARS_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
export PYTHONPATH="${_SETVARS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
